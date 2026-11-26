"""Renders every example to examples/out/*.pdf and docs/*.png (needs pdftoppm)."""
import pathlib, subprocess, runpy, sys

here = pathlib.Path(__file__).resolve().parent
(here / "out").mkdir(exist_ok=True)
docs = here.parent / "docs"
docs.mkdir(exist_ok=True)
for name in ("rich_text", "math", "showcase"):
    runpy.run_path(str(here / f"{name}.py"), run_name="__main__")
    subprocess.run(["pdftoppm", "-png", "-r", "110", "-singlefile",
                    str(here / "out" / f"{name}.pdf"), str(docs / name)], check=True)
    print("rendered", name, file=sys.stderr)
