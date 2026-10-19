# Setting up the shared workspace

The workspace lets a small group share models. Everyone keeps their own files on their own computer; the workspace
is a shared shelf on a server you run. People upload a model to it, and others browse and open what was shared. Every
upload is kept as a numbered **version**, so nothing is lost when two people work on the same model.

The server is [PocketBase](https://pocketbase.io): one small program with a built-in database, file storage and
accounts. It is free and runs on any spare computer. This guide was written against PocketBase **0.40.5**, which is
what the setup file in `server/pb_migrations` was tested with. Use that version, or read PocketBase's changelog before
using a newer one.

## 1. Set up the server (once)

Do this on the spare computer that stays on.

1. Download PocketBase 0.40.5 for your system from <https://github.com/pocketbase/pocketbase/releases> (on Windows:
   `pocketbase_0.40.5_windows_amd64.zip`) and unzip it into its own folder, for example `C:\ChronovoxServer`.
2. Copy the **contents of this project's `server` folder** into that folder, so that it holds `pocketbase.exe`, the
   `pb_migrations` and `pb_hooks` folders and `start_server.bat` side by side.
3. Make your own owner account (this is for the server's dashboard, not for the editor). In a terminal in that folder:

   ```
   pocketbase.exe superuser create you@example.com "a long password only you know"
   ```

4. Start the server by double-clicking `start_server.bat`, or run:

   ```
   pocketbase.exe serve --http=0.0.0.0:8090
   ```

   The first start creates the tables for models, versions and invite codes by itself. `0.0.0.0` lets other computers
   connect; without it the server only answers the computer it runs on. Windows may ask whether to allow it through the
   firewall. Say yes for private networks.
5. Open the dashboard at `http://127.0.0.1:8090/_/` and log in with the owner account from step 3.
6. Make an **invite code**. In the dashboard open the collection `invites`, choose *New record*, type a code of at least
   six characters (a few random words is good) and save. Give that code to the people you want to let in. Delete the
   record to stop anyone else using it; accounts that already exist are not affected.

### Already running an older setup?

Copy the new files from the project's `server` folder into the PocketBase folder (the `pb_hooks` folder and any
new file in `pb_migrations`) and restart the server. New migrations are applied by themselves, and existing models and
versions are left alone.

## 2. Keeping the history in check

Two safety rails keep the workspace from filling up with copies nobody needs.

- **Nothing changed, nothing uploaded.** Before uploading, the editor compares the model with what the server already
  has. If it is the same model, in the same state, it says so and uploads nothing. This is about the model itself, not
  the file, so saving the file again without changing anything does not create a version.
- **A limit on the history.** The server keeps only the newest **25** versions of each model. When a new version is
  uploaded, the oldest ones beyond the limit are removed for good. Version numbers keep counting up, so you may see
  versions 8 to 32 and nothing before them. The editor shows the limit and warns you, in red, right before an upload
  that would remove an old version.

To change the number, open the dashboard, the `workspace_settings` collection, edit its one record and change
`max_versions` (between 2 and 1000). It applies from the next upload. Lowering it does not delete anything at once;
older versions of a model go the next time somebody uploads to that model. Everyone can read the setting, but only you
can change it.

If a model has a version you must keep, download it and save the file somewhere safe, because it will eventually fall
off the end of the history.

## 3. Let your friends reach it

The server needs an address your friends' computers can connect to. Pick one.

- **Tailscale (recommended).** A free private network: you and each friend install the Tailscale app and join the same
  network, and the server PC gets a private address (like `100.x.y.z`) that only members can reach. Nothing is exposed
  to the public internet, and there is no router setup. The address to give your friends is that one plus the port,
  for example `100.64.0.5:8090`. Check Tailscale's current free-plan limits.
- **Same network.** If everyone is on the same local network, use the server PC's local address, for example
  `192.168.1.20:8090`.
- **Cloudflare Tunnel.** A public `https://` address on a domain you own, with no router setup and nothing for your
  friends to install. See the section below.
- **A public address, the hard way.** Port forwarding on your router with a name that follows your home address, and HTTPS in front
  of the server (for example with Caddy). It works, but it puts the server on the open internet, so it needs more care.
  **Do not use plain `http://` over the open internet:** passwords would travel unencrypted.

### Using a Cloudflare Tunnel

A tunnel gives the server a normal `https://models.yourdomain.com` address without opening anything on your router.
A small program (`cloudflared`) on the server PC makes an outgoing connection to Cloudflare, and Cloudflare passes
visitors through it. Cloudflare provides the HTTPS certificate, your home address stays hidden, and it keeps working
when your home address changes. Friends install nothing: they type the address into the editor.

Cloudflare's website changes from time to time, so if a menu has moved, follow their current "create a tunnel"
instructions. The idea stays the same.

**What you need**

- A domain name that uses Cloudflare for its DNS (the free plan is enough). You can buy one for roughly $10 a year, or
  add one you already have to Cloudflare. Without a domain only temporary tunnels are possible, with a random address
  that changes every time, which is no use for a permanent workspace.
- A free Cloudflare account.

**Setting it up**

1. Add your domain to Cloudflare (or buy one there) and wait until it shows as active.
2. In the Cloudflare dashboard open **Zero Trust**, then **Networks → Tunnels**, and create a tunnel of the
   *Cloudflared* type. Name it something like `chronovox`.
3. Choose Windows. Cloudflare shows an install command with your tunnel's secret in it. Run it in a terminal on the
   server PC, started as administrator. That installs `cloudflared` as a Windows service, so the tunnel starts with
   the computer.
4. Add a **public hostname** to the tunnel: subdomain `models` (or any name you like), your domain, service type
   `HTTP`, and the address `localhost:8090`.
5. Start PocketBase with **`start_server_tunnel.bat`** instead of `start_server.bat`. It answers only the server PC
   itself, so the tunnel is the only way in and nobody on your network can reach it directly.
6. Check it from any computer: open `https://models.yourdomain.com/api/health`. It should say the API is healthy.
7. In the editor, choose **Workspace → Workspace window...** and type `models.yourdomain.com` as the server address.
   Because it has no `http://` in front, the editor uses `https://`.

**Things to know**

- **The dashboard is public too.** `https://models.yourdomain.com/_/` is reachable by anyone, protected only by your
  owner password, so make it long and unique. In PocketBase's dashboard settings there is also a rate-limiting option,
  which is worth turning on. Cloudflare can additionally put its own login in front of just the `/_/` path (an
  *Access* application in Zero Trust) so only you get as far as the dashboard, while the editor's requests are left
  alone.
- **Cloudflare sees the traffic.** HTTPS ends at Cloudflare and starts again inside the tunnel, so Cloudflare could in
  principle read the models that pass through. For most people that is fine; it is worth knowing.
- **Size limits are no problem.** Cloudflare's free plan allows far larger uploads than the workspace's 4 MB.
- **If something is wrong,** an error page from Cloudflare usually means one of two things: the tunnel is not
  connected (check that the `cloudflared` service is running), or the tunnel works but PocketBase is not running (start
  `start_server_tunnel.bat`).
- **The editor says "Cloudflare blocked this request" (403).** Cloudflare sits in front of the server and can turn away
  traffic that does not look like a web browser. The editor identifies itself as `ChronovoxReforged/<version>`, which
  Cloudflare normally lets through. If it is still blocked, open the Cloudflare dashboard, go to **Security → Events**
  and look at the blocked request: it names the rule that stopped it. The usual culprits are **Browser Integrity
  Check** and **Bot Fight Mode** (both under **Security → Settings**). Turning one off for the whole domain works, and
  so does a *Configuration Rule* that switches Browser Integrity Check off only for requests whose URI path starts
  with `/api/`, which leaves the dashboard protected. Menus move around, so follow Cloudflare's current documentation.
- **The editor says "Cloudflare could not reach the workspace server" (502, 530 and similar).** The tunnel and
  Cloudflare are fine, but nothing answers behind them: start `start_server_tunnel.bat`, and check the `cloudflared`
  service is running on that computer.
- **Dashboard works but the editor login fails.** The dashboard (`/_/`) uses the *owner* account you made with
  `superuser create`. The editor uses a separate account made by **registering** with an invite code. The owner account
  does not work in the editor, and an editor account does not work in the dashboard.
- **Tailscale Funnel** is a similar idea without buying a domain: if you already use Tailscale it can publish the
  server at a public `https://….ts.net` address. It has its own limits, so check their current documentation.

## 4. Using it in the editor

1. Choose **Workspace → Workspace window...**.
2. Type the **server address** (for example `100.64.0.5:8090`). Pick **Register**, fill in your email, name, a password
   of at least 8 characters and the invite code, and choose *Create account*. Next time just **Log in**. The editor
   remembers your login, so you do not have to type it again.
3. **Share a model:** open it, then in *This model* give it a name and choose *Share as a new model*. The editor saves
   the file first if needed.
4. **Upload a new version:** after you change a shared model, choose *Upload as a new version*, with a short note if you
   like. The editor knows which shared model each file belongs to.
5. **Open someone's model:** expand it in *Shared models* and choose *Open the newest*, or *Open* on any older version.
   It is downloaded to `Documents\Chronovox Workspace` and opened in a new tab. Nothing is ever overwritten: if the
   name is taken you get `Name (v3).cvox` instead, so your own work-in-progress is always safe.
6. **If someone else uploaded first:** when you upload, the editor checks that the server's newest version is the one
   your file was based on. If not, it tells you and asks. *Upload mine anyway* makes yours the newest version, and
   theirs stays in the history. To keep both, open their version and copy matrices between tabs (Ctrl+C and Ctrl+V).

Everything else in the editor works without an account or a connection.

## 5. Looking after the server

- **Backups.** Everything lives in the `pb_data` folder next to the program. In the dashboard, *Settings → Backups* can
  make a backup file; copy those somewhere other than this computer from time to time.
- **Keeping it running.** The server only works while the program is running. To start it when the computer starts, use
  Windows Task Scheduler (*At startup*) or a tool such as NSSM.
- **Forgotten password.** The server cannot send email, so the "forgot password" link will not work. Open the dashboard,
  the `users` collection, edit the person and set a new password.
- **Removing someone.** Delete their record in the `users` collection. Models and versions they uploaded stay.
- **Updating.** Replace `pocketbase.exe` with a newer one. Read PocketBase's changelog first, because it is still before
  version 1.0 and can change things between releases, and back up first.
- **Disk space.** Models are small, and the history limit keeps the number of copies in check. A single model file can
  be at most 4 MB (most are a few KB), which the editor checks before it uploads. To allow more, change `maxSize` of the
  `file` field of the `versions` collection in the dashboard, and the same number in the editor
  (`MAX_FILE_BYTES` in `chronovox/workspace_api.py`).

## 6. Who can do what

- Anyone with an account can see every shared model and upload a new version to any of them. It is one shared workspace.
- Only the person who first shared a model can rename or delete it.
- An uploaded version can never be changed or deleted on its own. The only ways one goes away are the history limit
  and deleting the whole shared model (only its first owner or you in the dashboard can), which removes all of its
  versions.
- You can see and change everything from the dashboard.

## For developers

- The server rules are in `server/pb_migrations`, and the code that trims old versions is `server/pb_hooks/chronovox.pb.js`.
- `tests/test_workspace_server.py` starts a real PocketBase and checks the rules. It is skipped unless the
  `POCKETBASE_EXE` environment variable points at a PocketBase program. It also covers the history limit.
