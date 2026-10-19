@echo off
rem Starts the Chronovox workspace server for use behind a Cloudflare Tunnel (or anything else running on this same
rem computer). It only answers this computer, so nothing else on your network can reach it directly.
rem Keep this file next to pocketbase.exe and the pb_migrations and pb_hooks folders.
cd /d "%~dp0"
echo Chronovox workspace server (this computer only). Close this window to stop it.
pocketbase.exe serve --http=127.0.0.1:8090
pause
