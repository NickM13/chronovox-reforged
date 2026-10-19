@echo off
rem Starts the Chronovox workspace server. Keep this file next to pocketbase.exe and the pb_migrations folder.
cd /d "%~dp0"
echo Chronovox workspace server. Close this window to stop it.
pocketbase.exe serve --http=0.0.0.0:8090
pause
