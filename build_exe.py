"""Build a single Windows program, dist/ChronovoxReforged.exe, that can be sent to people.

    .venv\\Scripts\\python.exe build_exe.py

Needs PyInstaller (``pip install pyinstaller``). The result needs no Python install on the other computer. It does
need a graphics card and driver that support OpenGL 3.3, which any PC from the last ten years or so does.
"""
import shutil
import subprocess
import sys
from pathlib import Path

from chronovox import __version__, icon

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"


def main() -> None:
    BUILD.mkdir(exist_ok=True)
    ico = BUILD / "icon.ico"
    ico.write_bytes(icon.make_ico())                 # the picture Explorer shows for the .exe
    command = [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--onefile", "--windowed",
        "--name", "ChronovoxReforged",
        "--icon", str(ico),
        "--collect-all", "glfw",                     # the GLFW library file that glfw loads at start-up
        "--add-data", f"{ROOT / 'assets' / 'icon_256.png'};assets",     # the picture for the window itself
        "--distpath", str(ROOT / "dist"), "--workpath", str(BUILD / "work"), "--specpath", str(BUILD),
        str(ROOT / "run_chronovox.py"),
    ]
    print("Building Chronovox Reforged", __version__)
    subprocess.run(command, check=True, cwd=ROOT)
    exe = ROOT / "dist" / "ChronovoxReforged.exe"
    print(f"\nDone: {exe} ({exe.stat().st_size / 1_000_000:.0f} MB)")


if __name__ == "__main__":
    main()
