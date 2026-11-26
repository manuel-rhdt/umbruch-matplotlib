"""Tests against the real umbruch binary ($UMBRUCH or PATH); skipped without it."""
import io, os, re, shutil, subprocess, sys, tempfile
import xml.etree.ElementTree as ET
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("module://umbruch_matplotlib")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
import umbruch_matplotlib as hb

out = tempfile.mkdtemp()
have_umbruch = hb.get_bridge() is not None


def fig_with_everything():
    fig, ax = plt.subplots(figsize=(5, 3.5), layout="constrained")
    x = np.linspace(0, 1, 40)
    ax.plot(x, np.sin(6 * x), label=r"curve $\sin(6x)$")
    ax.set_title(hb.rich("<i>Rate</i> r<sub>0</sub> in <b>Hz</b>"))
    ax.set_xlabel("plain label")
    ax.legend()
    rng = np.random.default_rng(0)
    xs, ys = rng.random((2, 30))
    ins = ax.inset_axes([0.55, 0.12, 0.4, 0.35])
    ins.tripcolor(xs, ys, np.hypot(xs, ys), shading="gouraud")
    return fig


# 1. malformed rich markup is reported, whatever the bridge
class Dummy(hb.UmbruchBridge):
    def measure_many(self, items):
        for m, f in items:
            self._cache[self.key(m, f)] = (10.0, 8.0, 2.0)
            self._ascent[self.key(m, f)] = (9.0, 11.0)
    def render(self, svg, w, h):
        return b"%PDF-dummy"

saved, hb.config.bridge = hb.config.bridge, Dummy()
fig, ax = plt.subplots(); ax.set_title(hb.rich("<b>oops"))
try:
    fig.savefig(os.path.join(out, "bad.pdf")); raise SystemExit("expected ValueError")
except ValueError as e:
    print("malformed markup ->", str(e)[:50])
plt.close(fig)

# 2. gouraud modes (no renderer needed: SVG output)
for mode in ("subdivide", "native", "stock"):
    hb.config.gouraud = mode
    fig = fig_with_everything()
    p = os.path.join(out, f"{mode}.svg"); fig.savefig(p)
    t = open(p, encoding="utf-8").read(); ET.fromstring(t)
    assert ("data-colors" in t) == (mode == "native"), mode
    assert ("colorAdd" in t) == (mode == "stock"), mode
    assert "foreignObject" in t and "<i>Rate</i> r<sub>0</sub>" in t
    plt.close(fig)
hb.config.gouraud = "subdivide"
hb.config.bridge = saved

if not have_umbruch:
    print("umbruch not found: skipping the end-to-end tests"); sys.exit(0)

# 3. metrics: ink heights/descents come from umbruch's measure op and must agree with
# Matplotlib's own glyph extents to within 2% of the font size
bridge = hb.get_bridge()
assert isinstance(bridge, hb.UmbruchServer), type(bridge)
assert bridge.request('hello', {})[0]['protocol'] == 1
for s in ["0.5", "plain label", "Wavy yg"]:
    prop = FontProperties(size=10)
    w, h, d = bridge.measure(hb.escape(s), hb._font_spec(prop))
    w0, h0, d0 = TextToPath().get_text_width_height_descent(s, prop, False)
    assert abs(h - h0) < 0.2 and abs(d - d0) < 0.2 and 0.95 * w0 <= w <= 1.2 * w0, (s, w, w0, h, h0, d, d0)
w, h, d = bridge.measure("E = mc<sup>2</sup>", hb._font_spec(FontProperties(size=10)))
_, h0, _ = bridge.measure("E = mc2", hb._font_spec(FontProperties(size=10)))
assert h > h0, "a superscript must make the box taller"

bridge._cache.clear()

# 4. end to end: PDF, tight bbox, text is real text
fig = fig_with_everything()
pdf = os.path.join(out, "fig.pdf"); fig.savefig(pdf)
assert open(pdf, "rb").read(4) == b"%PDF"
fig.savefig(os.path.join(out, "tight.pdf"), bbox_inches="tight")
if shutil.which("pdftotext"):
    txt = subprocess.run(["pdftotext", pdf, "-"], capture_output=True, text=True).stdout
    for word in ("Rate", "plain label", "Hz"):
        assert word in txt, (word, txt)
# 5. the server survives a crash: kill it, the next request starts a new one
pid = bridge._proc.pid; bridge._proc.kill(); bridge._proc.wait()
bridge._cache.clear()
assert bridge.measure("again", hb._font_spec(FontProperties(size=10)))[0] > 0
assert bridge._proc.pid != pid
# batching: many probes in one request
bridge.measure_many([(f"label {i}", hb._font_spec(FontProperties(size=10))) for i in range(3000)])
print("ok", out)

# 6. unsupported features fail instead of falling back
import matplotlib.pyplot as plt
for label, kw in [(r"$\notamacro{x}$", {}), ("x", {"usetex": False})]:
    pass
fig, ax = plt.subplots(); ax.set_xlabel(r"$\definitelynotamacro{x}$")
try:
    fig.savefig(os.path.join(out, "bad.pdf")); raise SystemExit("unsupported math did not fail")
except RuntimeError as e:
    assert "MathML" in str(e), e
plt.close(fig)
print("failure tests ok")


def test_tick_label_exponents_use_the_text_font():
    import matplotlib.pyplot as plt, subprocess, tempfile
    fig, ax = plt.subplots()
    ax.loglog([1, 1e4], [1e-3, 1e3])
    path = os.path.join(tempfile.mkdtemp(), "log.pdf")
    fig.savefig(path)
    fonts = subprocess.run(["pdffonts", path], capture_output=True, text=True).stdout
    assert "STIX" not in fonts, fonts
    for bad in (r"\mathdefault{\alpha}",):
        try:
            hb._mathdefault_html(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("unsupported macro must raise")


def test_single_letter_mathrm_is_upright():
    import umbruch_matplotlib as u
    ml = u._default_tex_to_mathml(r"\mathrm{s}^{-1}")
    assert 'mathvariant="normal">s<' in ml, ml
    assert 'mathvariant="normal"' not in u._default_tex_to_mathml("s")


def test_math_font_is_set_on_math_elements():
    f = hb._set_math_font
    assert f("<math><mi>x</mi></math>", "Foo Math") == \
        "<math style=\"font-family:'Foo Math'\"><mi>x</mi></math>"
    out = f('<math display="block" style="color:red"><mi>x</mi></math>', "A, B")
    assert out.startswith('<math display="block" style="font-family:\'A\', \'B\';color:red">'), out


def test_math_font_changes_the_pdf_font():
    if not have_umbruch:
        return
    def fonts(family):
        hb.config.math_font = family
        try:
            fig = plt.figure(figsize=(2, 1))
            fig.text(0.1, 0.4, r"$\alpha^2+\frac{a}{b}$")
            path = os.path.join(out, "mathfont.pdf")
            fig.savefig(path)
            plt.close(fig)
        finally:
            hb.config.math_font = None
        return subprocess.run(["pdffonts", path], capture_output=True, text=True).stdout
    assert "LatinModern" in fonts("Latin Modern Math")
    assert "STIXTwo" in fonts("STIX Two Math")


if __name__ == "__main__":
    test_math_font_is_set_on_math_elements()
    test_math_font_changes_the_pdf_font()
    test_tick_label_exponents_use_the_text_font()
    test_single_letter_mathrm_is_upright()
