"""The workspace client against a real PocketBase server, started just for these tests.

Skipped unless PocketBase is available: set POCKETBASE_EXE to its path (or have ``pocketbase`` on the PATH).
It checks the rules in server/pb_migrations: invite-only sign-up, who may do what, and versions that cannot clash.
"""
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from chronovox import cvox
from chronovox import workspace_api as api

EXE = os.environ.get("POCKETBASE_EXE") or shutil.which("pocketbase")
pytestmark = pytest.mark.skipif(not EXE, reason="set POCKETBASE_EXE to run the server tests")
MIGRATIONS = Path(__file__).resolve().parent.parent / "server" / "pb_migrations"
HOOKS = Path(__file__).resolve().parent.parent / "server" / "pb_hooks"
ADMIN_TOKEN = {}
ADMIN, ADMIN_PASSWORD, INVITE = "owner@example.com", "a-long-test-password", "test-invite-code"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    data = tmp_path_factory.mktemp("pb_data")
    subprocess.run([EXE, "superuser", "upsert", ADMIN, ADMIN_PASSWORD, "--dir", str(data)],
                   check=True, capture_output=True)
    port = free_port()
    process = subprocess.Popen([EXE, "serve", f"--http=127.0.0.1:{port}", "--dir", str(data),
                                "--migrationsDir", str(MIGRATIONS), "--hooksDir", str(HOOKS)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            urllib.request.urlopen(url + "/api/health", timeout=1)
            break
        except OSError:
            time.sleep(0.1)
    else:
        process.kill()
        pytest.fail("PocketBase did not start")
    # the owner makes an invite code, as the guide describes
    login = urllib.request.Request(url + "/api/collections/_superusers/auth-with-password", headers={
        "Content-Type": "application/json"}, data=json.dumps({"identity": ADMIN, "password": ADMIN_PASSWORD}).encode())
    token = json.load(urllib.request.urlopen(login))["token"]
    ADMIN_TOKEN["token"] = token
    invite = urllib.request.Request(url + "/api/collections/invites/records", headers={
        "Content-Type": "application/json", "Authorization": token}, data=json.dumps({"code": INVITE}).encode())
    urllib.request.urlopen(invite)
    yield url
    process.kill()
    process.wait()


def account(server, name):
    client = api.WorkspaceApi(server)
    client.register(f"{name}@example.com", "pass12345678", name.title(), INVITE)
    return client, client.login(f"{name}@example.com", "pass12345678")


def small_model_bytes():
    from chronovox.viewer import demo_model
    return cvox.dumps(demo_model(), compress=True)


def test_registration_needs_a_valid_invite_code(server):
    client = api.WorkspaceApi(server)
    for invite in ("", "wrong-code"):
        with pytest.raises(api.ApiError):
            client.register("nobody@example.com", "pass12345678", "Nobody", invite)
    client.register("somebody@example.com", "pass12345678", "Somebody", INVITE)
    assert client.login("somebody@example.com", "pass12345678").label == "Somebody"
    assert client.refresh().email == "somebody@example.com"


def test_share_list_and_download_round_trip(server):
    client, me = account(server, "alice")
    data = small_model_bytes()
    model, first = client.share_new("Dragon", me, data, "Dragon.cvox", b"\x89PNG\r\n\x1a\nxx", "first")
    second = client.upload_version(model.id, 2, me.id, "second", data + b"", "Dragon.cvox")
    listed = next(m for m in client.list_models() if m.id == model.id)
    assert [v.number for v in listed.versions] == [2, 1] and listed.latest.author == "Alice"
    assert listed.versions[1].note == "first" and client.latest_number(model.id) == 2
    assert client.download_version(second) == data and client.download_preview(first).startswith(b"\x89PNG")
    assert client.download_preview(second) is None


def test_two_people_cannot_both_upload_the_same_version_number(server):
    alice, a = account(server, "carol")
    bob, b = account(server, "dave")
    data = small_model_bytes()
    model, _ = alice.share_new("Shared", a, data, "s.cvox", None)
    bob.upload_version(model.id, 2, b.id, "bob first", data, "s.cvox")
    with pytest.raises(api.ApiError) as lost:
        alice.upload_version(model.id, 2, a.id, "alice second", data, "s.cvox")
    assert lost.value.conflict
    assert [v.author for v in alice.list_versions(model.id)] == ["Dave", "Carol"]


def test_people_cannot_delete_other_peoples_models_or_forge_the_author(server):
    alice, a = account(server, "erin")
    bob, b = account(server, "frank")
    data = small_model_bytes()
    model, version = alice.share_new("Mine", a, data, "m.cvox", None)
    with pytest.raises(api.ApiError):
        bob.delete_model(model.id)
    with pytest.raises(api.ApiError):
        bob.upload_version(model.id, 2, a.id, "pretending to be Erin", data, "m.cvox")     # author is not the logged-in user
    assert [m.id for m in bob.list_models() if m.id == model.id] == [model.id]              # still there, and visible to all
    alice.delete_model(model.id)
    assert all(m.id != model.id for m in alice.list_models())


def test_nothing_works_without_logging_in(server):
    shared, me = account(server, "heidi")
    shared.share_new("Visible to members only", me, small_model_bytes(), "v.cvox", None)
    nobody = api.WorkspaceApi(server)
    assert nobody.list_models() == []                              # the server shows a visitor an empty list
    with pytest.raises(api.ApiError):
        nobody.create_model("Sneaky", me.id)


def set_limit(server, number):
    """The owner changes the number of versions kept, as in the dashboard."""
    request = urllib.request.Request(
        server + "/api/collections/workspace_settings/records/chronovoxconfig", method="PATCH",
        headers={"Content-Type": "application/json", "Authorization": ADMIN_TOKEN["token"]},
        data=json.dumps({"max_versions": number}).encode())
    urllib.request.urlopen(request)


def test_the_default_limit_is_25_and_people_can_read_but_not_change_it(server):
    client, me = account(server, "ivan")
    assert client.get_max_versions() == 25
    request = urllib.request.Request(
        server + "/api/collections/workspace_settings/records/chronovoxconfig", method="PATCH",
        headers={"Content-Type": "application/json", "Authorization": client.token},
        data=json.dumps({"max_versions": 2}).encode())
    with pytest.raises(urllib.error.HTTPError) as denied:
        urllib.request.urlopen(request)
    assert denied.value.code in (400, 403, 404) and client.get_max_versions() == 25


def test_only_the_newest_versions_are_kept_and_numbers_keep_counting(server):
    set_limit(server, 3)
    try:
        client, me = account(server, "judy")
        data = small_model_bytes()
        model, _ = client.share_new("Busy", me, data, "b.cvox", None)
        for number in range(2, 6):
            client.upload_version(model.id, number, me.id, f"edit {number}", data + bytes([number]), "b.cvox")
        kept = client.list_versions(model.id)
        assert [v.number for v in kept] == [5, 4, 3]                     # 1 and 2 were removed to stay at 3
        assert client.latest_number(model.id) == 5
        assert client.download_version(kept[0]) == data + bytes([5])     # what is kept is intact
        client.upload_version(model.id, 6, me.id, "one more", data + b"six", "b.cvox")
        assert [v.number for v in client.list_versions(model.id)] == [6, 5, 4]
    finally:
        set_limit(server, 25)


def test_the_limit_applies_to_each_model_separately(server):
    set_limit(server, 2)
    try:
        client, me = account(server, "kate")
        data = small_model_bytes()
        busy, _ = client.share_new("Busy2", me, data, "b.cvox", None)
        quiet, _ = client.share_new("Quiet", me, data, "q.cvox", None)
        for number in (2, 3, 4):
            client.upload_version(busy.id, number, me.id, "", data + bytes([number]), "b.cvox")
        assert [v.number for v in client.list_versions(busy.id)] == [4, 3]
        assert [v.number for v in client.list_versions(quiet.id)] == [1]
    finally:
        set_limit(server, 25)


def test_the_newest_version_check_sees_the_checksum_of_the_file(server):
    client, me = account(server, "leo")
    data = small_model_bytes()
    model, _ = client.share_new("Checked", me, data, "c.cvox", None)
    assert client.latest_version(model.id).sha256 == api.sha256_hex(data)


def raw_upload(server, client, model_id, author_id, number, size):
    """Upload a file straight to the server, past the editor's own size check, to see what the server does."""
    body, content_type = api.encode_multipart(
        {"model": model_id, "number": str(number), "author": author_id, "note": "", "sha256": ""},
        {"file": ("big.cvox", b"x" * size, "application/octet-stream")})
    request = urllib.request.Request(server + "/api/collections/versions/records", data=body, method="POST",
                                     headers={"Content-Type": content_type, "Authorization": client.token})
    return urllib.request.urlopen(request).status


def test_the_server_itself_refuses_files_over_4_mb(server):
    client, me = account(server, "mallory")
    model, _ = client.share_new("Sizes", me, small_model_bytes(), "s.cvox", None)
    assert raw_upload(server, client, model.id, me.id, 2, 4 * 1024 * 1024) == 200           # exactly the limit is fine
    with pytest.raises(urllib.error.HTTPError) as refused:
        raw_upload(server, client, model.id, me.id, 3, 4 * 1024 * 1024 + 1)
    assert refused.value.code == 400
    assert client.latest_number(model.id) == 2
