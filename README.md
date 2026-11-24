# helium-matplotlib

A Matplotlib backend that draws figures as SVG and renders them to PDF with
[helium](https://github.com/manuel-rhdt/helium). Text and math are laid out by helium:
labels may contain rich text (`rich("<i>t</i><sub>0</sub>")`) and `$...$` math is converted to
MathML. Needs a helium binary with `--serve` (set `$HELIUM` or put it on `PATH`).

```python
import matplotlib
matplotlib.use("module://helium_backend")
import matplotlib.pyplot as plt
from helium_backend import rich

fig, ax = plt.subplots()
ax.set_xlabel(rich("time <i>t</i> (ms)"))
ax.set_title(r"$\hat\rho_{ij}=\frac{a}{b}$")
fig.savefig("fig.pdf")
```

Install with `pip install .[math]` (the `math` extra pulls in `latex2mathml`).

Unsupported features raise instead of falling back to Matplotlib's own text rendering:
`text.usetex`, and TeX that cannot be converted to MathML.

Defaults: DejaVu families are mapped to the bundled TeX Gyre fonts (`config.font_map`),
Gouraud triangles are subdivided until helium reports `mesh_gouraud`.

Tests: `HELIUM=/path/to/helium python tests/test_helium_backend.py`.
