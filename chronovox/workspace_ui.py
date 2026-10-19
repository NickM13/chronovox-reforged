"""The Workspace window: log in to the shared server, share the current model, browse and open what others shared.

Network calls run on a small thread pool so the editor never freezes while it waits; the results are picked up on
the main thread once per frame (``poll``), which is the only place that changes the editor.
"""
from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

from imgui_bundle import imgui

from . import png
from . import workspace_files as wf
from . import workspace_store as ws
from .workspace_api import ApiError, ModelVersion, SharedModel, User, WorkspaceApi, normalize_url, sha256_hex

REFRESH_SECONDS = 30.0
ERROR_COLOR = imgui.ImVec4(1.0, 0.45, 0.4, 1.0)
GOOD_COLOR = imgui.ImVec4(0.55, 0.9, 0.55, 1.0)


class StaleError(Exception):
    """The server has a newer version than the one this file is based on. ``latest`` is None if not known."""

    def __init__(self, latest: int | None) -> None:
        super().__init__("a newer version exists")
        self.latest = latest


class NothingNewError(Exception):
    """The file is identical to a version that is on the server already, so there is nothing to upload."""

    def __init__(self, number: int) -> None:
        super().__init__(f"identical to version {number}")
        self.number = number


class WorkspacePanel:
    def __init__(self, viewer, folder: Path | None, url: str = "") -> None:
        """``folder``: where the login and file links are kept (the preferences folder); None keeps nothing on disk."""
        self.viewer = viewer
        self.folder = folder
        self.links = ws.LinkStore(ws.links_path(folder) if folder else None)
        self.session = ws.load_session(ws.session_path(folder) if folder else None)
        self.api = WorkspaceApi(self.session.url, self.session.token) if self.session else None
        self.open = False
        self.url_text = self.session.url if self.session else url
        self.mode = "login"                       # "login" or "register"
        self.email_text = self.session.email if self.session else ""
        self.password_text = ""
        self._reveal_password = False                          # the Show button was held down last frame
        self.name_text = ""
        self.invite_text = ""
        self.note_text = ""
        self.share_name = ""
        self.models: list[SharedModel] | None = None
        self.max_versions: int | None = None                   # how many versions per model the server keeps
        self.message: tuple[str, bool] = ("", False)          # text, is it an error
        self.conflict: dict | None = None                     # an upload waiting for the person's decision
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="workspace")
        self._jobs: list[tuple[Future, object, object, str]] = []
        self._last_refresh = 0.0
        self._sha_cache: tuple | None = None
        if self.session:
            self._run(self.api.refresh, self._session_ok, self._session_failed, "Checking your login")

    # -- background work ---------------------------------------------------------------------------
    @property
    def busy(self) -> str:
        """What the program is waiting for, or an empty string."""
        return self._jobs[0][3] if self._jobs else ""

    def _run(self, work, on_ok, on_error=None, label: str = "Working") -> None:
        self._jobs.append((self._pool.submit(work), on_ok, on_error, label))

    def poll(self) -> None:
        """Hand finished results to their handlers. Call once per frame."""
        for job in [j for j in self._jobs if j[0].done()]:
            self._jobs.remove(job)
            future, on_ok, on_error, _ = job
            try:
                result = future.result()
            except Exception as error:                       # noqa: BLE001 - every failure ends up as a message
                if on_error is not None and on_error(error):
                    continue
                self._fail(error)
            else:
                on_ok(result)
        if (self.open and self.session and self.models is not None and not self._jobs and self.conflict is None
                and time.perf_counter() - self._last_refresh > REFRESH_SECONDS):
            self.refresh()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _say(self, text: str, error: bool = False) -> None:
        self.message = (text, error)

    def _fail(self, error: Exception) -> None:
        if isinstance(error, ApiError) and error.unauthorized and self.session:
            self._forget_login()
            self._say("Your login has expired. Please log in again.", True)
        else:
            self._say(str(error) or error.__class__.__name__, True)

    # -- account -----------------------------------------------------------------------------------
    def _session_ok(self, user: User) -> None:
        self._remember(user)

    def _session_failed(self, error: Exception) -> bool:
        if isinstance(error, ApiError) and error.unreachable:
            self._say(str(error), True)             # keep the login: the server may just be off for now
            return True
        return False

    def _remember(self, user: User) -> None:
        self.session = ws.Session(self.api.base, self.api.token, user.id, user.email, user.name)
        ws.save_session(self.session, ws.session_path(self.folder) if self.folder else None)
        self.url_text = self.api.base

    def _forget_login(self) -> None:
        self.session = self.api = self.models = None
        self.conflict = None
        ws.clear_session(ws.session_path(self.folder) if self.folder else None)

    def log_in(self) -> None:
        api = WorkspaceApi(self.url_text)
        email, password = self.email_text, self.password_text

        def done(user: User) -> None:
            self.api, self.password_text = api, ""
            self._remember(user)
            self._say(f"Logged in as {user.label}.")
            self.refresh()

        self._run(lambda: api.login(email, password), done, None, "Logging in")

    def register(self) -> None:
        api = WorkspaceApi(self.url_text)
        email, password, name, invite = self.email_text, self.password_text, self.name_text, self.invite_text

        def work() -> User:
            api.register(email, password, name, invite)
            return api.login(email, password)

        def done(user: User) -> None:
            self.api, self.password_text, self.invite_text = api, "", ""
            self._remember(user)
            self._say(f"Welcome, {user.label}. Your account is ready and you are logged in.")
            self.refresh()

        def failed(error: Exception) -> bool:
            if isinstance(error, ApiError) and error.status == 400:
                self._say("Could not register. Check the invite code, that the email is not used already, "
                          "and that the password has at least 8 characters.", True)
                return True
            return False

        self._run(work, done, failed, "Creating your account")

    def log_out(self) -> None:
        self._forget_login()
        self._say("Logged out.")

    # -- shared models -----------------------------------------------------------------------------
    def refresh(self) -> None:
        if not self.api:
            return
        self._last_refresh = time.perf_counter()
        api = self.api

        def work():
            return api.list_models(), api.get_max_versions()

        self._run(work, self._got_models, None, "Loading the shared models")

    def _got_models(self, result) -> None:
        self.models, self.max_versions = result
        self._last_refresh = time.perf_counter()

    def _latest_known(self, model_id: str) -> int | None:
        for model in self.models or []:
            if model.id == model_id:
                return model.latest.number if model.latest else None
        return None

    def open_version(self, model: SharedModel, version: ModelVersion) -> None:
        """Get a version of a shared model into a file and open it in the editor."""
        if not self.api:
            return
        is_latest = model.latest is not None and model.latest.number == version.number
        folder = self.folder_for_downloads()
        try:
            path, there = wf.choose_download_path(folder, model.name, version.number, is_latest, self.links,
                                                  self.session.url, model.id)
        except OSError as error:
            self._say(str(error), True)
            return
        if there:                                   # that very version is in the folder already and untouched
            self.viewer.open_file(path)
            self._say(f"Opened {path.name}.")
            return
        api = self.api

        def done(content: bytes) -> None:
            if version.sha256 and sha256_hex(content) != version.sha256:
                self._say("The download arrived damaged. Please try again.", True)
                return
            target, exists = wf.choose_download_path(folder, model.name, version.number, is_latest, self.links,
                                                     self.session.url, model.id)
            if not exists:
                try:
                    folder.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                except OSError as error:
                    self._say(f"Could not save {target.name}: {error}", True)
                    return
            self.links.set(target, ws.Link(self.session.url, model.id, model.name, version.number,
                                           sha256_hex(wf.canonical(content))))
            self.viewer.open_file(target)
            self._say(f"Opened version {version.number} of {model.name} as {target.name}.")

        self._run(lambda: api.download_version(version), done, None, f"Downloading {model.name}")

    def folder_for_downloads(self) -> Path:
        return wf.default_folder()

    # -- sharing the current model -----------------------------------------------------------------
    def _preview_png(self) -> bytes | None:
        """A small picture of the model being shared, for the list. Failing to make one never stops an upload."""
        try:
            pixels, texture = self.viewer._render_preview(self.viewer.model)
            texture.release()
            return png.encode_png(pixels[::-1])
        except Exception:                            # noqa: BLE001
            return None

    def upload_current(self, force: bool = False) -> None:
        """Share the open model: as a new shared model, or as the next version of the one it is linked to."""
        v = self.viewer
        if not self.api or v.model is None:
            return
        if v.path is None or v.dirty:                # what is uploaded is exactly the file on disk
            v.save()
            if v.path is None or v.dirty:
                self._say("Save the model first, then it can be shared.", True)
                return
        path = v.path
        content = wf.canonical(path.read_bytes())            # what is uploaded and compared is the model, not its encoding
        link = self.links.get(path, self.session.url)
        if link is not None and sha256_hex(content) == link.sha256:          # no need to ask the server
            self._say(f"Nothing has changed since version {link.version}, so nothing was uploaded.")
            return
        preview, note, api = self._preview_png(), self.note_text.strip(), self.api
        user = User(self.session.user_id, self.session.email, self.session.name)
        name = self.share_name.strip() or path.stem

        def work():
            if link is None:
                return api.share_new(name, user, content, path.name, preview, note)
            newest = api.latest_version(link.model_id)
            latest = newest.number if newest else 0
            if newest is not None and newest.sha256 == sha256_hex(content):  # somebody already uploaded exactly this
                raise NothingNewError(newest.number)
            if latest != link.version and not force:
                raise StaleError(latest)
            number = latest + 1
            try:
                version = api.upload_version(link.model_id, number, user.id, note, content, path.name, preview)
            except ApiError as error:
                if error.conflict:
                    raise StaleError(None) from None
                raise
            return SharedModel(link.model_id, link.model_name, "", "", ""), version

        def done(result) -> None:
            model, version = result
            self.links.set(path, ws.Link(self.session.url, model.id, model.name, version.number, sha256_hex(content)))
            self.conflict = None
            self.note_text = ""
            self.share_name = ""
            self._say(f"Shared {model.name} as version {version.number}.")
            self.refresh()

        def failed(error: Exception) -> bool:
            if isinstance(error, StaleError):
                self.conflict = {"latest": error.latest}
                self._say("")
                return True
            if isinstance(error, NothingNewError):
                self.conflict = None
                self._say(f"This is identical to version {error.number} on the server, so nothing was uploaded.")
                return True
            return False

        self._run(work, done, failed, "Uploading")

    # -- drawing -----------------------------------------------------------------------------------
    def draw(self) -> None:
        """The contents of the Workspace tab of the sidebar. ``open`` says whether that tab is showing."""
        if imgui.begin_child("##workspace_body"):
            imgui.push_item_flag(imgui.ItemFlags_.no_tab_stop, True)     # Tab visits the text fields, see _text
            if self.busy:
                imgui.text_disabled(self.busy + "...")
            if self.message[0]:
                imgui.push_text_wrap_pos(0.0)
                imgui.text_colored(ERROR_COLOR if self.message[1] else GOOD_COLOR, self.message[0])
                imgui.pop_text_wrap_pos()
            imgui.begin_disabled(bool(self.busy))
            if self.session is None:
                self._draw_login()
            else:
                self._draw_account()
            imgui.end_disabled()
            imgui.pop_item_flag()
        imgui.end_child()

    @staticmethod
    def _text(label: str, value: str, flags: int = 0, hint: str = "") -> tuple[bool, str]:
        """A text field that the Tab key stops at (everything else in this window is skipped by Tab)."""
        imgui.push_item_flag(imgui.ItemFlags_.no_tab_stop, False)
        result = (imgui.input_text_with_hint(label, hint, value, flags) if hint
                  else imgui.input_text(label, value, flags))
        imgui.pop_item_flag()
        return result

    def _draw_login(self) -> None:
        imgui.text_wrapped("Log in to share models with the people you work with, and to open what they have shared. "
                           "Everything else in the editor works without an account.")
        imgui.separator()
        enter = imgui.InputTextFlags_.enter_returns_true
        _, self.url_text = self._text("Server", self.url_text, hint="e.g. 100.64.0.5:8090 or models.example.com")
        for mode, label in (("login", "Log in"), ("register", "Register")):
            if imgui.radio_button(f"{label}##mode", self.mode == mode):
                self.mode = mode
            imgui.same_line()
        imgui.new_line()
        _, self.email_text = self._text("Email", self.email_text)
        if self.mode == "register":
            _, self.name_text = self._text("Name", self.name_text)
        # The password is hidden unless the Show button next to it is held down (it is hidden again on release).
        style = imgui.get_style()
        button_width = imgui.calc_text_size("Show").x + 2 * style.frame_padding.x
        imgui.set_next_item_width(imgui.calc_item_width() - button_width - style.item_spacing.x)
        hide = 0 if self._reveal_password else imgui.InputTextFlags_.password
        submitted, self.password_text = self._text("##password", self.password_text, hide | enter)
        imgui.same_line()
        imgui.button("Show##password")
        self._reveal_password = imgui.is_item_active()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Hold down to see the password")
        imgui.same_line()
        imgui.text("Password")
        if self.mode == "register":
            invite_done, self.invite_text = self._text("Invite code", self.invite_text, enter)
            submitted = submitted or invite_done
            imgui.text_disabled("Ask the person who runs the server for an invite code.")
            imgui.text_disabled("Passwords need at least 8 characters.")
        ready = bool(normalize_url(self.url_text) and self.email_text.strip() and self.password_text)
        if self.mode == "register":
            ready = ready and bool(self.invite_text.strip())
        imgui.begin_disabled(not ready)
        if imgui.button("Log in" if self.mode == "login" else "Create account") or (submitted and ready):
            self._say("")
            self.log_in() if self.mode == "login" else self.register()
        imgui.end_disabled()

    def _draw_account(self) -> None:
        s = self.session
        imgui.text(f"Logged in as {s.name or s.email}")
        imgui.same_line()
        if imgui.small_button("Log out"):
            self.log_out()
            return
        imgui.same_line()
        if imgui.small_button("Refresh"):
            self.refresh()
        imgui.text_disabled(s.url)
        imgui.separator()
        self._draw_this_model()
        imgui.separator()
        self._draw_shared_list()

    def _current_sha(self, path: Path) -> str | None:
        """The checksum of the file on disk, recomputed only when the file changes."""
        try:
            info = path.stat()
        except OSError:
            return None
        key = (str(path), info.st_mtime_ns, info.st_size)
        if self._sha_cache is None or self._sha_cache[0] != key:
            self._sha_cache = (key, wf.file_sha(path))
        return self._sha_cache[1]

    def _draw_this_model(self) -> None:
        v = self.viewer
        imgui.text("This model")
        if v.model is None:
            imgui.text_disabled("Open or create a model to share it.")
            return
        path = v.path
        link = self.links.get(path, self.session.url) if path else None
        latest = self._latest_known(link.model_id) if link else None
        status = wf.link_status(link, self._current_sha(path) if path else None, latest)
        imgui.text_wrapped(status + ("." if link else ""))
        if v.dirty and link:
            imgui.text_disabled("There are unsaved changes in the editor. Uploading saves them first.")
        if self.conflict is not None:
            self._draw_conflict(link)
            return
        if link is None:
            _, self.share_name = self._text(
                "Name", self.share_name, hint=path.stem if path else "name of the shared model")
        _, self.note_text = self._text("Note", self.note_text, hint="what changed (optional)")
        unchanged = link is not None and not v.dirty and self._current_sha(path) == link.sha256
        imgui.begin_disabled(unchanged)
        if imgui.button("Share as a new model" if link is None else "Upload as a new version"):
            self.upload_current()
        imgui.end_disabled()
        if unchanged:
            imgui.same_line()
            imgui.text_disabled("No changes to upload")
        self._draw_limit_note(link)
        if link is not None and latest is not None and latest > link.version:
            imgui.same_line()
            model = next((m for m in self.models or [] if m.id == link.model_id), None)
            if model is not None and imgui.button(f"Open version {latest}"):
                self.open_version(model, model.latest)

    def _draw_limit_note(self, link) -> None:
        """Say how many versions the server keeps, and warn when the next upload would remove old ones."""
        if not self.max_versions:
            return
        model = next((m for m in self.models or [] if link and m.id == link.model_id), None)
        dropped = wf.versions_to_drop([x.number for x in model.versions], self.max_versions) if model else []
        text = f"The server keeps the newest {self.max_versions} versions of each model."
        if dropped:
            which = f"version {dropped[0]}" if len(dropped) == 1 else f"versions {dropped[0]} to {dropped[-1]}"
            imgui.push_text_wrap_pos(0.0)
            imgui.text_colored(ERROR_COLOR, f"{text} Uploading now will remove {which} for good.")
            imgui.pop_text_wrap_pos()
        else:
            imgui.text_disabled(text)

    def _draw_conflict(self, link) -> None:
        latest = self.conflict.get("latest")
        based = link.version if link else "?"
        text = (f"Someone has uploaded version {latest} since the version this file is based on ({based})."
                if latest else "Someone uploaded a newer version just now.")
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(ERROR_COLOR, text + " Uploading yours will become the newest version, and theirs stays "
                                              "in the history.")
        imgui.pop_text_wrap_pos()
        if imgui.button("Upload mine anyway"):
            self.upload_current(force=True)
        imgui.same_line()
        if imgui.button("Cancel"):
            self.conflict = None
        imgui.same_line()
        if imgui.button("Refresh the list"):
            self.refresh()

    def _draw_shared_list(self) -> None:
        imgui.text("Shared models")
        if self.models is None:
            if not self.busy:
                self.refresh()
            imgui.text_disabled("Loading...")
            return
        if not self.models:
            imgui.text_disabled("Nothing has been shared yet.")
            return
        imgui.begin_child("##shared", imgui.ImVec2(0, 0), 0, imgui.WindowFlags_.no_nav_inputs)
        pending = None
        for model in self.models:
            latest = model.latest
            heading = f"{model.name}"
            if latest:
                heading += f"   v{latest.number} by {latest.author}, {wf.pretty_time(latest.created)}"
            opened = imgui.tree_node(f"{heading}##model{model.id}")
            if opened:
                if latest and imgui.small_button(f"Open the newest##open{model.id}"):
                    pending = (model, latest)
                for version in model.versions:
                    imgui.push_id(version.id)
                    imgui.text(f"v{version.number}  {version.author}  {wf.pretty_time(version.created)}")
                    if version.note:
                        imgui.same_line()
                        imgui.text_disabled(version.note)
                    imgui.same_line()
                    if imgui.small_button("Open"):
                        pending = (model, version)
                    imgui.pop_id()
                imgui.tree_pop()
        imgui.end_child()
        if pending:
            self.open_version(*pending)
