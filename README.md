# umbruch-matplotlib

*Umbruch* (OOM-brookh) is German for page makeup, and also for upheaval.

A Matplotlib backend that draws figures as SVG and renders them to PDF with
[umbruch](https://github.com/manuel-rhdt/umbruch). Text and math are laid out by umbruch:
labels may contain rich text (`rich("<i>t</i><sub>0</sub>")`) and `$...$` math is converted to
MathML. The umbruch executable comes with the `umbruch` package; `$UMBRUCH` or `PATH` override it.

```python
import matplotlib
matplotlib.use("module://umbruch_matplotlib")
import matplotlib.pyplot as plt
from umbruch_matplotlib import rich

fig, ax = plt.subplots()
ax.set_xlabel(rich("time <i>t</i> (ms)"))
ax.set_title(r"$\hat\rho_{ij}=\frac{a}{b}$")
fig.savefig("fig.pdf")
```

Install with `pip install .[math]` (the `math` extra pulls in `latex2mathml`).

Unsupported features raise instead of falling back to Matplotlib's own text rendering:
`text.usetex`, and TeX that cannot be converted to MathML.

Fonts come from Matplotlib: for every text, the font file that `font_manager.findfont` picks (including fonts added with `fontManager.addfont`) is sent to umbruch as an `@font-face` with a `data:` URI, so the PDF uses the face Matplotlib measured, with real italic and bold faces, and nothing depends on umbruch seeing the system's fonts. The families of the text follow as CSS fallbacks. The cost is the font file per request (a few milliseconds for 600 kB). Variable fonts are drawn in their default instance, and from a `.ttc` only the first face is read.

Tick labels with exponents (`10^3`, `1×10^-4`) are set as text with superscripts in the label font, not as math. Math in `$...$` is converted to MathML and set in STIX Two Math by default. `mathtext.fontset = "cm"` selects Latin Modern Math, and `umbruch_matplotlib.config.math_font` takes the family of any OpenType math font that Matplotlib finds (for example `"Libertinus Math"`, sent to umbruch like the text fonts) or the path of a font file; it needs a font with an OpenType MATH table. Gouraud triangles are subdivided until umbruch reports `mesh_gouraud`.

### Wide-gamut colours

Matplotlib has no colour management: every RGB value (colours, colormaps, image pixels) is three numbers in [0, 1]. Say which space they are in, and umbruch writes the PDF in that space:

```python
umbruch_matplotlib.config.color_space = "display-p3"   # or fig.savefig(..., color_space="display-p3")
ax.plot(x, y, color=(0.1, 0.9, 0.3))                   # these numbers are Display P3
```

The spaces are those of CSS Color 4: `srgb` (the default), `display-p3`, `a98-rgb`, `prophoto-rgb` and `rec2020`. Painted colours (fill, stroke, text) are written as `color(display-p3 r g b)` from Matplotlib's float values, without the rounding to 1/255 of stock SVG. Hatch patterns and Matplotlib's stock Gouraud gradient stops still go through 8-bit hex, and raster images are 8 bits per channel as Matplotlib makes them.

Everything Matplotlib itself supplies as sRGB (names like `"red"`, hex colours, the default colour cycle, the built-in colormaps) is then misread as P3 numbers. Convert it: `convert_color("tab:blue", dst="display-p3")` and `convert_cmap("viridis", dst="display-p3")` return values in the target space, clipped to its gamut.

Raster images (`imshow`, rasterised artists) get an ICC profile of the space in their PNG. Today's umbruch ignores the profile of an image in SVG, so images are still read as sRGB; the tests skip that check until it does. Mixed-up colours between flat and image are thus possible for now. The native Gouraud mesh markup (`data-colors`) is in the figure's space too, and umbruch will have to be told so.

## Examples

Rich text through `rich()`: inline markup for italics, bold, sub- and superscripts, colour and size, in titles, labels, tick labels and annotations.

![Rich text](docs/rich_text.png)

Math is laid out as MathML with STIX Two Math, with correct operator spacing and function application, and can be mixed with rich text.

![Math](docs/math.png)

A larger figure with hatching, gradients, colour bars, annotations, and math and rich text in titles, labels and legends. The output is a vector PDF whose text stays selectable.

![Showcase](docs/showcase.png)

The scripts are in `examples/`. `python examples/make_images.py` re-renders the PNGs (needs `pdftoppm` and an `umbruch` binary on `$UMBRUCH` or the path).

Tests: `pytest tests` (the ones that need the engine use the `umbruch` package, `$UMBRUCH` or the path, and are skipped without it).

## License

MIT, see `LICENSE`.
