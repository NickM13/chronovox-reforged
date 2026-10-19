"""The workspace client and the local store, tested against a small fake server so no real one is needed."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from chronovox import workspace_api as api
from chronovox import workspace_store as store


# --- addresses and request bodies -----------------------------------------------------------------

@pytest.mark.parametrize("typed, expected", [
    ("  http://nas.example.com:8090/ ", "http://nas.example.com:8090"),
    ("https://models.example.com", "https://models.example.com"),
    ("localhost:8090", "http://localhost:8090"),
    ("192.168.1.20:8090", "http://192.168.1.20:8090"),
    ("10.0.0.5", "http://10.0.0.5"),
    ("100.101.102.103:8090", "http://100.101.102.103:8090"),           # a Tailscale address
    ("studio-pc.tail1234.ts.net:8090", "http://studio-pc.tail1234.ts.net:8090"),
    ("mypc", "http://mypc"),
    ("models.example.com", "https://models.example.com"),
    ("8.8.8.8:8090", "https://8.8.8.8:8090"),
    ("", ""),
])
def test_normalize_url(typed, expected):
    assert api.normalize_url(typed) == expected


def test_multipart_body_has_every_field_and_file_and_closes_properly():
    body, content_type = api.encode_multipart({"model": "abc", "note": "héllo"}, {"file": ("a.cvox", b"\x00\x01binary", "x/y")})
    boundary = content_type.split("boundary=")[1].encode()
    assert body.startswith(b"--" + boundary) and body.endswith(b"--" + boundary + b"--\r\n")
    assert b'name="model"\r\n\r\nabc\r\n' in body and "héllo".encode() in body
    assert b'name="file"; filename="a.cvox"' in body and b"Content-Type: x/y\r\n\r\n\x00\x01binary\r\n" in body


def test_sha256_hex():
    assert api.sha256_hex(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_an_empty_upload_is_refused_before_anything_is_sent():
    client = api.WorkspaceApi("http://127.0.0.1:1")
    with pytest.raises(api.ApiError, match="nothing to upload"):
        client.upload_version("m", 1, "u", "", b"", "a.cvox")


# --- a fake server ------------------------------------------------------------------------------

class Fake(BaseHTTPRequestHandler):
    log: list = []
    agents: list = []
    responses: dict = {}            # (method, path) -> (status, payload) or (status, payload, extra headers)

    def _answer(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        Fake.log.append((self.command, self.path, self.headers.get("Authorization"), body))
        Fake.agents.append(self.headers.get("User-Agent"))
        key = (self.command, self.path.split("?")[0])
        reply = Fake.responses.get(key, (404, {"message": "not found"}))
        status, payload = reply[:2]
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        for name, value in (reply[2] if len(reply) > 2 else {}).items():
            self.send_header(name, value)
        if len(reply) > 2 and reply[2].get("Content-Type"):
            pass
        else:
            self.send_header("Content-Type", "application/octet-stream" if isinstance(payload, bytes) else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_DELETE = _answer

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    Fake.log, Fake.agents, Fake.responses = [], [], {}
    httpd = HTTPServer(("127.0.0.1", 0), Fake)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}"
    httpd.shutdown()
    httpd.server_close()


USER = {"id": "u1", "email": "alice@example.com", "name": "Alice"}


def test_login_keeps_the_token_and_sends_it_on_later_requests(server):
    Fake.responses[("POST", "/api/collections/users/auth-with-password")] = (200, {"token": "T1", "record": USER})
    Fake.responses[("POST", "/api/collections/users/auth-refresh")] = (200, {"token": "T2", "record": USER})
    client = api.WorkspaceApi(server)
    user = client.login(" alice@example.com ", "secret")
    assert user == api.User("u1", "alice@example.com", "Alice") and user.label == "Alice" and client.token == "T1"
    assert json.loads(Fake.log[0][3]) == {"identity": "alice@example.com", "password": "secret"}
    assert Fake.log[0][2] is None                                   # no token while logging in
    client.refresh()
    assert Fake.log[1][2] == "T1" and client.token == "T2"


def test_register_sends_the_invite_code_and_confirms_the_password(server):
    Fake.responses[("POST", "/api/collections/users/records")] = (200, USER)
    api.WorkspaceApi(server).register("alice@example.com", "secret123", " Alice ", " code-1 ")
    sent = json.loads(Fake.log[0][3])
    assert sent["invite"] == "code-1" and sent["passwordConfirm"] == "secret123" and sent["name"] == "Alice"


def test_server_errors_become_readable_messages(server):
    Fake.responses[("POST", "/api/collections/users/auth-with-password")] = (
        400, {"message": "Failed to authenticate.", "data": {"identity": {"message": "Invalid value."}}})
    with pytest.raises(api.ApiError) as caught:
        api.WorkspaceApi(server).login("a@b.c", "x")
    assert caught.value.status == 400 and "Failed to authenticate." in str(caught.value) and "Invalid value." in str(caught.value)


def test_an_unreachable_server_is_reported_as_such():
    with pytest.raises(api.ApiError) as caught:
        api.WorkspaceApi("http://127.0.0.1:1", timeout=2).login("a@b.c", "x")
    assert caught.value.unreachable and "Cannot reach the server" in str(caught.value)


def test_a_reply_that_is_not_json_is_explained(server):
    Fake.responses[("POST", "/api/collections/users/auth-refresh")] = (200, b"<html>hello</html>")
    with pytest.raises(api.ApiError, match="not a workspace reply"):
        api.WorkspaceApi(server, token="T").refresh()


def version_record(number, model="m1", author="u1", created="2026-10-16 10:00:00.000Z", **extra):
    return {"id": f"v{number}{model}", "model": model, "number": number, "author": author, "note": f"note {number}",
            "created": created, "sha256": "aa", "file": f"f{number}.cvox", "preview": "p.png",
            "expand": {"author": {"id": author, "name": "Alice", "email": "alice@example.com"}}, **extra}


def test_list_models_groups_versions_newest_first_and_orders_models_by_latest_upload(server):
    Fake.responses[("GET", "/api/collections/models/records")] = (200, {"page": 1, "totalPages": 1, "items": [
        {"id": "m1", "name": "Dragon", "owner": "u1", "updated": "2026-01-01", "expand": {"owner": {"name": "Alice"}}},
        {"id": "m2", "name": "Tree", "owner": "u2", "updated": "2026-02-01", "expand": {"owner": {"email": "bob@x.com"}}},
        {"id": "m3", "name": "Empty", "owner": "u2", "updated": "2026-03-01"}]})
    Fake.responses[("GET", "/api/collections/versions/records")] = (200, {"page": 1, "totalPages": 1, "items": [
        version_record(1, "m1", created="2026-10-01 00:00:00.000Z"), version_record(2, "m1", created="2026-10-02 00:00:00.000Z"),
        version_record(1, "m2", created="2026-10-10 00:00:00.000Z"), version_record(9, "gone")]})
    models = api.WorkspaceApi(server, "T").list_models()
    assert [m.name for m in models] == ["Tree", "Dragon", "Empty"]          # by newest upload, then by model date
    dragon = models[1]
    assert [v.number for v in dragon.versions] == [2, 1] and dragon.latest.number == 2
    assert dragon.owner == "Alice" and models[0].owner == "bob" and dragon.latest.author == "Alice"
    assert models[2].latest is None


def test_list_requests_follow_every_page(server):
    pages = {1: {"page": 1, "totalPages": 2, "items": [{"id": "m1", "name": "A", "owner": "u"}]},
             2: {"page": 2, "totalPages": 2, "items": [{"id": "m2", "name": "B", "owner": "u"}]}}
    seen = []

    class Paged(Fake):
        def _answer(self):
            page = int(self.path.split("page=")[1].split("&")[0])
            seen.append(page)
            data = json.dumps(pages[page] if "/models/" in self.path else {"page": 1, "totalPages": 1, "items": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        do_GET = _answer

    httpd = HTTPServer(("127.0.0.1", 0), Paged)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        models = api.WorkspaceApi(f"http://127.0.0.1:{httpd.server_port}", "T").list_models()
    finally:
        httpd.shutdown()
        httpd.server_close()
    assert [m.name for m in models] == ["A", "B"] and 1 in seen and 2 in seen


def test_a_lost_race_for_a_version_number_is_flagged_as_a_conflict(server):
    Fake.responses[("POST", "/api/collections/versions/records")] = (400, {"message": "Failed to create record."})
    Fake.responses[("GET", "/api/collections/versions/records")] = (200, {"page": 1, "totalPages": 1, "items": [{"number": 4}]})
    client = api.WorkspaceApi(server, "T")
    with pytest.raises(api.ApiError) as caught:
        client.upload_version("m1", 4, "u1", "", b"data", "a.cvox")
    assert caught.value.conflict
    Fake.responses[("GET", "/api/collections/versions/records")] = (200, {"page": 1, "totalPages": 1, "items": [{"number": 3}]})
    with pytest.raises(api.ApiError) as caught:                    # a different 400 (number 4 is free): not a conflict
        client.upload_version("m1", 4, "u1", "", b"data", "a.cvox")
    assert not caught.value.conflict


def test_download_gets_a_file_pass_first_then_the_file(server):
    Fake.responses[("POST", "/api/files/token")] = (200, {"token": "PASS"})
    Fake.responses[("GET", "/api/files/versions/v1/f1.cvox")] = (200, b"CVOX-bytes")
    version = api.ModelVersion("v1", "m1", 1, "u1", "Alice", "", "", "", "f1.cvox", "")
    client = api.WorkspaceApi(server, "T")
    assert client.download_version(version) == b"CVOX-bytes"
    assert Fake.log[-1][1].endswith("?token=PASS") and Fake.log[-1][2] is None          # the pass, not the login, is sent
    assert client.download_preview(version) is None                                       # this version has no preview


def test_a_failed_first_upload_removes_the_empty_model(server):
    Fake.responses[("POST", "/api/collections/models/records")] = (200, {"id": "m9", "name": "New", "owner": "u1"})
    Fake.responses[("POST", "/api/collections/versions/records")] = (400, {"message": "Failed to create record."})
    Fake.responses[("GET", "/api/collections/versions/records")] = (200, {"page": 1, "totalPages": 1, "items": []})
    Fake.responses[("DELETE", "/api/collections/models/records/m9")] = (204, {})
    with pytest.raises(api.ApiError):
        api.WorkspaceApi(server, "T").share_new("New", api.User("u1", "a@b.c", "A"), b"data", "n.cvox", None)
    assert ("DELETE", "/api/collections/models/records/m9") in [(c, p) for c, p, _, _ in Fake.log]


# --- local store ---------------------------------------------------------------------------------

def test_the_session_round_trips_and_a_bad_file_means_none(tmp_path):
    path = tmp_path / "s.json"
    assert store.load_session(path) is None
    session = store.Session("http://pc:8090", "TOKEN", "u1", "a@b.c", "Alice")
    assert store.save_session(session, path) and store.load_session(path) == session
    path.write_text("{broken")
    assert store.load_session(path) is None
    path.write_text(json.dumps({"url": "x", "token": "", "user_id": "u", "email": "e"}))
    assert store.load_session(path) is None                       # an empty token is no login
    store.save_session(session, path)
    store.clear_session(path)
    assert store.load_session(path) is None and not path.exists()


def test_links_remember_a_local_file_per_server(tmp_path):
    links = store.LinkStore(tmp_path / "links.json")
    model = tmp_path / "Dragon.cvox"
    link = store.Link("http://pc:8090", "m1", "Dragon", 3, "abc")
    assert links.get(model, "http://pc:8090") is None
    links.set(model, link)
    assert links.get(model, "http://pc:8090") == link
    assert links.get(model, "http://other") is None               # another server's models are not this one's
    again = store.LinkStore(tmp_path / "links.json")              # and it is kept on disk
    assert again.get(model, "http://pc:8090") == link
    again.remove(model)
    assert store.LinkStore(tmp_path / "links.json").get(model, "http://pc:8090") is None


# --- file names, dates and link status ----------------------------------------------------------

from chronovox import workspace_files as wf


def test_safe_name_removes_characters_a_file_name_cannot_have():
    assert wf.safe_name('Dragon: "v2"?') == "Dragon_ _v2__"
    assert wf.safe_name(r"a/b\c") == "a_b_c" and wf.safe_name("  Tree. ") == "Tree" and wf.safe_name("...") == "model"


def test_pretty_time_converts_utc_to_a_short_local_time_and_leaves_junk_alone():
    shown = wf.pretty_time("2026-10-16 18:40:02.858Z")
    assert len(shown) == 16 and shown.startswith("2026-10-1")
    assert wf.pretty_time("yesterday") == "yesterday"


def test_download_names_never_overwrite_and_reuse_an_unchanged_copy(tmp_path):
    links = store.LinkStore()
    server = "http://pc:8090"
    path, there = wf.choose_download_path(tmp_path, "Dragon", 3, True, links, server, "m1")
    assert path.name == "Dragon.cvox" and not there
    path.write_bytes(b"first copy")
    links.set(path, store.Link(server, "m1", "Dragon", 3, api.sha256_hex(b"first copy")))
    # the same version again, untouched: use the file that is there
    assert wf.choose_download_path(tmp_path, "Dragon", 3, True, links, server, "m1") == (path, True)
    # a newer version: the old file is somebody's work, so it gets a new name
    newer, there = wf.choose_download_path(tmp_path, "Dragon", 4, True, links, server, "m1")
    assert newer.name == "Dragon (v4).cvox" and not there
    newer.write_bytes(b"x")
    # edited since it was downloaded: not reused either, and the name after that is numbered
    path.write_bytes(b"my edits")
    again, there = wf.choose_download_path(tmp_path, "Dragon", 3, True, links, server, "m1")
    assert again.name == "Dragon (v3).cvox" and not there
    # an older version always carries its number
    assert wf.choose_download_path(tmp_path, "Tree", 2, False, links, server, "m2")[0].name == "Tree (v2).cvox"


def test_link_status_describes_each_situation():
    link = store.Link("s", "m1", "Dragon", 3, "aa")
    assert wf.link_status(None, None, None) == "Not shared yet"
    assert wf.link_status(link, "aa", 3) == "Linked to Dragon, version 3, up to date"
    assert wf.link_status(link, "bb", 3) == "Linked to Dragon, version 3, changed here since then"
    assert wf.link_status(link, "aa", 5) == "Linked to Dragon, version 3, version 5 is on the server"
    assert wf.link_status(link, "bb", 5).endswith("changed here since then, version 5 is on the server")
    assert wf.link_status(link, "aa", None) == "Linked to Dragon, version 3"                 # server not asked yet


# --- safety rails: the identical-file check and the version limit ----------------------------------

def test_versions_to_drop_lists_the_oldest_that_the_next_upload_pushes_out():
    assert wf.versions_to_drop([3, 1, 2], 5) == []
    assert wf.versions_to_drop([1, 2, 3, 4], 5) == []                  # 5 after the upload: still room
    assert wf.versions_to_drop([1, 2, 3, 4, 5], 5) == [1]
    assert wf.versions_to_drop([5, 3, 4, 1, 2], 3) == [1, 2, 3]       # a lower limit than before drops several
    assert wf.versions_to_drop([1, 2, 3], None) == [] and wf.versions_to_drop([1, 2, 3], 0) == []


def test_the_server_limit_is_read_and_a_missing_or_nonsense_setting_means_none(server):
    path = ("GET", "/api/collections/workspace_settings/records/chronovoxconfig")
    client = api.WorkspaceApi(server, "T")
    Fake.responses[path] = (200, {"id": "chronovoxconfig", "max_versions": 12})
    assert client.get_max_versions() == 12
    Fake.responses[path] = (404, {"message": "not found"})              # a server set up before this setting existed
    assert client.get_max_versions() is None
    Fake.responses[path] = (200, {"max_versions": 0})
    assert client.get_max_versions() is None
    Fake.responses[path] = (401, {"message": "login needed"})
    with pytest.raises(api.ApiError):
        client.get_max_versions()                                      # but a login problem is still reported


def test_latest_version_gives_the_newest_with_its_checksum_or_none(server):
    path = ("GET", "/api/collections/versions/records")
    client = api.WorkspaceApi(server, "T")
    Fake.responses[path] = (200, {"page": 1, "totalPages": 1, "items": [version_record(7)]})
    newest = client.latest_version("m1")
    assert newest.number == 7 and newest.sha256 == "aa"                # version_record fixes the checksum text
    Fake.responses[path] = (200, {"page": 1, "totalPages": 1, "items": []})
    assert client.latest_version("m1") is None


def test_canonical_bytes_do_not_depend_on_how_a_model_file_was_written():
    import numpy as np
    from chronovox import cvox
    from chronovox.model import Matrix, Model
    palette = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], np.uint8)
    voxels = np.ones((2, 2, 2), np.uint16)                                     # uses only the first color
    model = Model(palette, [Matrix("A", voxels)])
    plain, packed = cvox.dumps(model), cvox.dumps(model, compress=True)
    assert plain != packed                                                      # written two ways...
    assert wf.canonical(plain) == wf.canonical(packed)                          # ...the same model
    changed = Model(palette, [Matrix("A", np.where(np.arange(8).reshape(2, 2, 2) == 3, 2, 1).astype(np.uint16))])
    assert wf.canonical(cvox.dumps(changed)) != wf.canonical(plain)             # a different model differs
    assert wf.canonical(b"not a cvox file") == b"not a cvox file"


def test_file_sha_is_about_the_model_in_the_file(tmp_path):
    import numpy as np
    from chronovox import cvox
    from chronovox.model import Matrix, Model
    model = Model(np.array([[1, 2, 3, 255]], np.uint8), [Matrix("A", np.ones((2, 2, 2), np.uint16))])
    a, b = tmp_path / "a.cvox", tmp_path / "b.cvox"
    cvox.save(model, a)
    cvox.save(model, b, compress=True)
    assert wf.file_sha(a) == wf.file_sha(b) is not None and wf.file_sha(tmp_path / "missing.cvox") is None


def test_a_file_over_the_size_limit_is_refused_before_anything_is_sent():
    client = api.WorkspaceApi("http://127.0.0.1:1")                      # nothing listens here: a request would say so
    with pytest.raises(api.ApiError, match="4 MB") as caught:
        client.upload_version("m", 1, "u", "", b"x" * (api.MAX_FILE_BYTES + 1), "big.cvox")
    assert "4.0 MB" in str(caught.value) and "Cannot reach" not in str(caught.value)


# --- behind Cloudflare -----------------------------------------------------------------------------

LOGIN = ("POST", "/api/collections/users/auth-with-password")
HTML = {"Content-Type": "text/html", "Server": "cloudflare", "cf-ray": "8abc123-AMS"}


def test_requests_say_who_they_are_instead_of_the_default_python_name(server):
    Fake.responses[LOGIN] = (200, {"token": "T", "record": USER})
    api.WorkspaceApi(server).login("a@b.c", "x")
    assert Fake.agents[-1].startswith("ChronovoxReforged/") and "Python-urllib" not in Fake.agents[-1]


def test_a_cloudflare_block_page_is_explained_and_marked(server):
    Fake.responses[LOGIN] = (403, b"<html>Error 1010: Access denied</html>", HTML)
    with pytest.raises(api.ApiError) as caught:
        api.WorkspaceApi(server).login("a@b.c", "x")
    error = caught.value
    assert error.status == 403 and error.cloudflare and not error.unreachable
    assert "Cloudflare blocked" in str(error) and "Security" in str(error) and "8abc123-AMS" in str(error)


def test_cloudflare_saying_it_cannot_reach_the_server_is_explained_differently(server):
    Fake.responses[LOGIN] = (530, b"<html>Error 1033</html>", HTML)
    with pytest.raises(api.ApiError) as caught:
        api.WorkspaceApi(server).login("a@b.c", "x")
    assert caught.value.cloudflare and "could not reach" in str(caught.value) and "tunnel" in str(caught.value)


def test_a_real_answer_from_the_workspace_server_is_not_mistaken_for_cloudflare(server):
    json_headers = {"Content-Type": "application/json", "cf-ray": "8abc123-AMS", "Server": "cloudflare"}
    Fake.responses[LOGIN] = (400, {"message": "Failed to authenticate.", "data": {}}, json_headers)
    with pytest.raises(api.ApiError) as caught:
        api.WorkspaceApi(server).login("a@b.c", "wrong")
    assert not caught.value.cloudflare and "Failed to authenticate." in str(caught.value)


def test_an_error_page_from_something_else_keeps_the_plain_message(server):
    Fake.responses[LOGIN] = (403, b"<html>nope</html>", {"Content-Type": "text/html"})
    with pytest.raises(api.ApiError) as caught:
        api.WorkspaceApi(server).login("a@b.c", "x")
    assert not caught.value.cloudflare and str(caught.value) == "The server answered 403."
