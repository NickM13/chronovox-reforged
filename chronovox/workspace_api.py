"""Talking to the shared workspace: a PocketBase server that holds accounts and shared models.

Plain standard library (urllib), no GL or UI imports, so it can run on a worker thread and be tested on its own.
The server side is set up by ``server/pb_migrations`` (see docs/WORKSPACE_SETUP.md).

Data on the server:
    users     accounts (register with an invite code)
    models    a shared model's name and first owner
    versions  one uploaded copy of a model: number, author, note, the .cvox file and a preview picture;
              never changed once uploaded, and (model, number) is unique, so two people cannot both upload v4
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field

from . import __version__

USER_AGENT = f"ChronovoxReforged/{__version__} (workspace client)"
CLOUDFLARE_DOWN = (502, 521, 522, 523, 524, 530)     # Cloudflare's own "cannot reach the server behind me" statuses

DEFAULT_TIMEOUT = 20.0
PAGE_SIZE = 200
MAX_FILE_BYTES = 4 * 1024 * 1024       # the largest model file the workspace takes (the server enforces the same)


class ApiError(Exception):
    """A request that did not work. ``status`` is the HTTP status (0 if the server could not be reached);
    ``conflict`` is set on an upload that lost to somebody else's upload of the same version number."""

    def __init__(self, message: str, status: int = 0, data: dict | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.data = data or {}
        self.conflict = False
        self.cloudflare = False            # the error page came from Cloudflare and not from the workspace server

    @property
    def unreachable(self) -> bool:
        return self.status == 0

    @property
    def unauthorized(self) -> bool:
        return self.status in (401, 403)


@dataclass(frozen=True)
class User:
    id: str
    email: str
    name: str

    @property
    def label(self) -> str:
        return self.name or self.email.split("@")[0]


@dataclass(frozen=True)
class ModelVersion:
    id: str
    model_id: str
    number: int
    author_id: str
    author: str          # display name
    note: str
    created: str         # as the server writes it: "2026-10-16 18:40:02.858Z"
    sha256: str
    file: str            # stored file name, needed to download it
    preview: str         # stored preview picture name, "" if none


@dataclass
class SharedModel:
    id: str
    name: str
    owner_id: str
    owner: str
    updated: str
    versions: list[ModelVersion] = field(default_factory=list)      # newest first

    @property
    def latest(self) -> ModelVersion | None:
        return self.versions[0] if self.versions else None


def normalize_url(text: str) -> str:
    """Turn what a person typed into the server's base address.

    Without http:// or https:// a local address (a name without dots, localhost, a private IP address) gets
    http:// and anything else gets https://. A trailing slash is dropped.
    """
    text = text.strip().rstrip("/")
    if not text:
        return ""
    if re.match(r"^https?://", text, re.I):
        return text
    host = text.split("/")[0].split(":")[0].lower()
    private = (host == "localhost" or "." not in host or host.endswith((".local", ".lan", ".ts.net"))
               or re.match(r"^(10\.|127\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.)", host))
    return ("http://" if private else "https://") + text


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]) -> tuple[bytes, str]:
    """A multipart/form-data body. ``files`` maps a field name to (file name, bytes, content type)."""
    boundary = "----chronovox" + uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode("utf-8"))
    for name, (filename, content, content_type) in files.items():
        parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
                      f"Content-Type: {content_type}\r\n\r\n").encode("utf-8") + content + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("ascii"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _behind_cloudflare(headers) -> bool:
    return bool(headers.get("cf-ray")) or (headers.get("Server") or "").lower() == "cloudflare"


def _cloudflare_message(status: int, headers) -> str:
    ray = headers.get("cf-ray") or ""
    suffix = f" (Cloudflare ray {ray})" if ray else ""
    if status in CLOUDFLARE_DOWN:
        return (f"Cloudflare could not reach the workspace server (error {status}). Check that the server and the "
                f"Cloudflare tunnel are both running.{suffix}")
    return (f"Cloudflare blocked this request (error {status}), so it never reached the workspace server. The server's "
            f"owner can see why under Security > Events in the Cloudflare dashboard; the setup guide explains what to "
            f"change.{suffix}")


def _message(payload, status: int) -> str:
    """What to tell a person about an error reply from the server."""
    if isinstance(payload, dict):
        details = []
        for key, value in (payload.get("data") or {}).items():
            if isinstance(value, dict) and value.get("message"):
                details.append(f"{key}: {value['message']}")
        text = payload.get("message") or f"The server answered {status}."
        return text + ((" " + " ".join(details)) if details else "")
    return f"The server answered {status}."


class WorkspaceApi:
    def __init__(self, base_url: str, token: str = "", timeout: float = DEFAULT_TIMEOUT) -> None:
        self.base = normalize_url(base_url)
        self.token = token
        self.timeout = timeout

    # -- plumbing --------------------------------------------------------------------------------
    def _request(self, method: str, path: str, *, body=None, raw: bytes | None = None, content_type: str | None = None,
                 query: dict | None = None, auth: bool = True, as_json: bool = True):
        if not self.base:
            raise ApiError("No server address is set.")
        url = self.base + path + (("?" + urllib.parse.urlencode(query)) if query else "")
        headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
        data = raw
        if body is not None:
            data, content_type = json.dumps(body).encode("utf-8"), "application/json"
        if content_type:
            headers["Content-Type"] = content_type
        if auth and self.token:
            headers["Authorization"] = self.token
        request = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as reply:
                content = reply.read()
        except urllib.error.HTTPError as error:
            payload = None
            try:
                payload = json.loads(error.read() or b"null")
            except ValueError:
                pass
            if payload is None and _behind_cloudflare(error.headers) and error.code >= 400:
                # an error page from Cloudflare itself (not an answer from the workspace server)
                failure = ApiError(_cloudflare_message(error.code, error.headers), error.code)
                failure.cloudflare = True
                raise failure from None
            raise ApiError(_message(payload, error.code), error.code,
                           payload.get("data") if isinstance(payload, dict) else None) from None
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            reason = getattr(error, "reason", error)
            raise ApiError(f"Cannot reach the server at {self.base} ({reason}).") from None
        if not as_json:
            return content
        try:
            return json.loads(content) if content else {}
        except ValueError:
            raise ApiError("The server sent something that is not a workspace reply. Is the address right?") from None

    def _all(self, path: str, **query) -> list[dict]:
        """Every record of a list request, page by page."""
        items, page = [], 1
        while True:
            reply = self._request("GET", path, query={**query, "page": page, "perPage": PAGE_SIZE})
            items += reply.get("items", [])
            if page >= int(reply.get("totalPages", 1) or 1):
                return items
            page += 1

    # -- accounts --------------------------------------------------------------------------------
    @staticmethod
    def _user(record: dict) -> User:
        return User(record["id"], record.get("email", ""), record.get("name", ""))

    def register(self, email: str, password: str, name: str, invite: str) -> User:
        """Create an account. The invite code is one the server's owner made."""
        record = self._request("POST", "/api/collections/users/records", auth=False, body={
            "email": email.strip(), "password": password, "passwordConfirm": password,
            "name": name.strip(), "invite": invite.strip()})
        return self._user(record)

    def login(self, email: str, password: str) -> User:
        reply = self._request("POST", "/api/collections/users/auth-with-password", auth=False,
                              body={"identity": email.strip(), "password": password})
        self.token = reply["token"]
        return self._user(reply["record"])

    def refresh(self) -> User:
        """Check the saved login still works and extend it."""
        reply = self._request("POST", "/api/collections/users/auth-refresh")
        self.token = reply["token"]
        return self._user(reply["record"])

    # -- shared models ---------------------------------------------------------------------------
    @staticmethod
    def _version(record: dict) -> ModelVersion:
        author = (record.get("expand") or {}).get("author") or {}
        return ModelVersion(
            id=record["id"], model_id=record["model"], number=int(record["number"]), author_id=record.get("author", ""),
            author=author.get("name") or (author.get("email", "").split("@")[0]) or "someone",
            note=record.get("note", ""), created=record.get("created", ""), sha256=record.get("sha256", ""),
            file=record.get("file", ""), preview=record.get("preview", ""))

    def list_models(self) -> list[SharedModel]:
        """Every shared model with all of its versions (newest first), the most recently changed model first."""
        models = {}
        for record in self._all("/api/collections/models/records", sort="-updated", expand="owner"):
            owner = (record.get("expand") or {}).get("owner") or {}
            models[record["id"]] = SharedModel(record["id"], record["name"], record.get("owner", ""),
                                               owner.get("name") or owner.get("email", "").split("@")[0],
                                               record.get("updated", ""))
        for record in self._all("/api/collections/versions/records", sort="-number", expand="author"):
            if record["model"] in models:
                models[record["model"]].versions.append(self._version(record))
        for model in models.values():
            model.versions.sort(key=lambda v: -v.number)
        # the model changed most recently first: by its newest upload, since uploads do not touch the model record
        return sorted(models.values(), key=lambda m: m.latest.created if m.latest else m.updated, reverse=True)

    def list_versions(self, model_id: str) -> list[ModelVersion]:
        records = self._all("/api/collections/versions/records", sort="-number", expand="author",
                            filter=f'model="{model_id}"')
        return [self._version(r) for r in records]

    def latest_version(self, model_id: str) -> ModelVersion | None:
        """The newest version of a model, or None if it has none."""
        reply = self._request("GET", "/api/collections/versions/records", query={
            "filter": f'model="{model_id}"', "sort": "-number", "perPage": 1, "page": 1, "expand": "author"})
        items = reply.get("items", [])
        return self._version(items[0]) if items else None

    def get_max_versions(self) -> int | None:
        """How many versions of a model the server keeps, or None if it does not say (an older server setup)."""
        try:
            record = self._request("GET", "/api/collections/workspace_settings/records/chronovoxconfig")
        except ApiError as error:
            if error.unreachable or error.unauthorized:
                raise
            return None
        value = record.get("max_versions")
        return int(value) if isinstance(value, (int, float)) and value >= 2 else None

    def latest_number(self, model_id: str) -> int:
        """The highest version number of a model, 0 if it has none."""
        reply = self._request("GET", "/api/collections/versions/records", query={
            "filter": f'model="{model_id}"', "sort": "-number", "perPage": 1, "page": 1, "fields": "number"})
        items = reply.get("items", [])
        return int(items[0]["number"]) if items else 0

    def upload_version(self, model_id: str, number: int, author_id: str, note: str, content: bytes,
                       file_name: str, preview_png: bytes | None = None) -> ModelVersion:
        """Add version ``number`` to a model. Raises ApiError with ``conflict`` set if that number is taken."""
        if not content:
            raise ApiError("There is nothing to upload.")
        if len(content) > MAX_FILE_BYTES:
            raise ApiError(f"This model is {len(content) / 1_048_576:.1f} MB, and the workspace takes files up to "
                           f"{MAX_FILE_BYTES // 1_048_576} MB.")
        files = {"file": (file_name, content, "application/octet-stream")}
        if preview_png:
            files["preview"] = ("preview.png", preview_png, "image/png")
        body, content_type = encode_multipart({
            "model": model_id, "number": str(number), "author": author_id, "note": note,
            "sha256": sha256_hex(content)}, files)
        try:
            record = self._request("POST", "/api/collections/versions/records", raw=body, content_type=content_type,
                                   query={"expand": "author"})
        except ApiError as error:
            if error.status == 400 and self.latest_number(model_id) >= number:
                error.conflict = True                      # somebody uploaded this number first
            raise
        return self._version(record)

    def create_model(self, name: str, owner_id: str) -> SharedModel:
        record = self._request("POST", "/api/collections/models/records", body={"name": name.strip(), "owner": owner_id})
        return SharedModel(record["id"], record["name"], record.get("owner", ""), "", record.get("updated", ""))

    def delete_model(self, model_id: str) -> None:
        self._request("DELETE", f"/api/collections/models/records/{model_id}", as_json=False)

    def share_new(self, name: str, owner: User, content: bytes, file_name: str, preview_png: bytes | None,
                  note: str = "") -> tuple[SharedModel, ModelVersion]:
        """Create a shared model and upload its first version. If the upload fails nothing is left behind."""
        model = self.create_model(name, owner.id)
        try:
            return model, self.upload_version(model.id, 1, owner.id, note, content, file_name, preview_png)
        except ApiError:
            try:
                self.delete_model(model.id)
            except ApiError:
                pass
            raise

    def download_version(self, version: ModelVersion) -> bytes:
        """The .cvox file of a version."""
        return self._download(version, version.file)

    def download_preview(self, version: ModelVersion) -> bytes | None:
        if not version.preview:
            return None
        return self._download(version, version.preview)

    def _download(self, version: ModelVersion, name: str) -> bytes:
        token = self._request("POST", "/api/files/token").get("token", "")       # files are protected: a short pass
        path = f"/api/files/versions/{version.id}/{urllib.parse.quote(name)}"
        return self._request("GET", path, query={"token": token}, as_json=False, auth=False)
