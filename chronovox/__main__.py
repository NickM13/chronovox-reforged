"""Run the viewer: python -m chronovox [file.cvox] [--screenshot out.png]"""
import argparse

from .preferences import default_path
from .viewer import Viewer, demo_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="chronovox", description="Chronovox Reforged")
    parser.add_argument("file", nargs="?", help=".cvox file to open (default: the home page)")
    parser.add_argument("--demo", action="store_true", help="start with the built-in demo model")
    parser.add_argument("--no-prefs", action="store_true", help="ignore and do not write saved preferences")
    parser.add_argument("--screenshot", metavar="PNG", help="render one frame to a PNG and exit")
    parser.add_argument("--preview", action="store_true",
                        help="start in preview (picture) mode: just the model, no panels, grid or outlines")
    args = parser.parse_args()

    use_prefs = not (args.no_prefs or args.screenshot)
    viewer = Viewer(visible=not args.screenshot, preferences_path=default_path() if use_prefs else None)
    if args.file:
        viewer.open_file(args.file)
    if viewer.model is None and args.demo:
        viewer.set_model(demo_model())     # without a file or --demo the viewer shows its home page
    if args.preview:
        viewer.set_preview(True)
    if args.screenshot:
        viewer.screenshot(args.screenshot)
        viewer.close()
    else:
        viewer.run()


if __name__ == "__main__":
    main()
