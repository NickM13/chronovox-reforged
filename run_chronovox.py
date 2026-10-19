"""Start Chronovox Reforged. This is the entry point of the built .exe (and works from source too).

A program built without a console has nowhere to print an error, so a crash is written to ``crash.log`` in the
preferences folder and shown in a message box instead of the program vanishing.
"""
import sys
import traceback


def report_crash() -> None:
    text = traceback.format_exc()
    try:
        from chronovox.preferences import default_path
        log = default_path().with_name("crash.log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text, encoding="utf-8")
        where = f"\n\nThe details were saved to:\n{log}"
    except Exception:                      # noqa: BLE001 - reporting must not fail in turn
        where = ""
    try:
        import tkinter
        from tkinter import messagebox
        root = tkinter.Tk()
        root.withdraw()
        messagebox.showerror("Chronovox Reforged stopped", text.strip().splitlines()[-1] + where)
        root.destroy()
    except Exception:                      # noqa: BLE001
        sys.stderr and sys.stderr.write(text)


if __name__ == "__main__":
    try:
        from chronovox.__main__ import main
        main()
    except SystemExit:
        raise
    except BaseException:                  # noqa: BLE001
        report_crash()
        sys.exit(1)
