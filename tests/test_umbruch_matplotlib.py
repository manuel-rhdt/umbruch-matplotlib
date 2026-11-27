"""Tests; those that need the real umbruch binary ($UMBRUCH, the package or PATH) are
skipped without it.  Run with ``pytest tests``."""
import io
import os
import re
import shutil
import subprocess
import sys
import zlib
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest
import umbruch
import matplotlib
matplotlib.use("module://umbruch_matplotlib")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
import umbruch_matplotlib as hb

try:
    hb.get_bridge()
    _have_umbruch = True
except umbruch.Error:
    _have_umbruch = False
need_umbruch = pytest.mark.skipif(not _have_umbruch, reason="umbruch not found")


@pytest.fixture(autouse=True)
def default_font():
    """A bundled family, whatever the matplotlibrc of the machine says."""
    with matplotlib.rc_context({"font.family": "DejaVu Sans"}):
        yield


class Dummy(hb.UmbruchBridge):
    def measure_many(self, items):
        for m, f in items:
            self._cache[self.key(m, f)] = (10.0, 8.0, 2.0)
            self._ascent[self.key(m, f)] = (9.0, 11.0)

    def render(self, svg, w, h):
        return b"%PDF-dummy"


@pytest.fixture
def dummy_bridge():
    saved, saved_g = hb.config.bridge, hb.config.gouraud
    hb.config.bridge = Dummy()
    yield hb.config.bridge
    hb.config.bridge, hb.config.gouraud = saved, saved_g
    plt.close("all")


@pytest.fixture
def real_bridge():
    bridge = hb.get_bridge()
    bridge._cache.clear()
    yield bridge
    plt.close("all")


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


# -- no renderer needed ---------------------------------------------------------

def test_malformed_rich_markup_is_reported(dummy_bridge, tmp_path):
    fig, ax = plt.subplots()
    ax.set_title(hb.rich("<b>oops"))
    with pytest.raises(ValueError, match="well-formed"):
        fig.savefig(tmp_path / "bad.pdf")


def _mesh_svg(bridge_caps, tmp_path, mode="auto", **save):
    hb.config.gouraud = mode
    bridge = hb.config.bridge
    bridge.capabilities = frozenset(bridge_caps)
    fig = fig_with_everything()
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", **save)
    plt.close(fig)
    svg = buf.getvalue().decode()
    ET.fromstring(svg)
    assert "foreignObject" in svg and "<i>Rate</i> r<sub>0</sub>" in svg
    return svg


def test_gouraud_subdivide_writes_flat_triangles(dummy_bridge, tmp_path):
    for caps, mode in (({"mesh_gouraud"}, "subdivide"), (set(), "auto")):
        svg = _mesh_svg(caps, tmp_path, mode)
        assert "data-umbruch-mesh" not in svg and svg.count("<path") > 100


def test_gouraud_native_writes_the_compact_mesh(dummy_bridge, tmp_path):
    for mode in ("auto", "native"):
        svg = _mesh_svg({"mesh_gouraud"}, tmp_path, mode)
        m = re.search(r'data-umbruch-mesh="gouraud" data-vertices="([^"]*)" '
                      r'data-triangles="([^"]*)"', svg)
        v, tri = (np.array(m.group(i).split(), dtype=float) for i in (1, 2))
        assert v.size % 6 == 0 and tri.size % 3 == 0 and tri.max() < v.size // 6
        assert v.size // 6 < tri.size       # the vertices of neighbours are shared
        assert (v[2:][::6] >= 0).all() and (v[5::6] == 1).all()     # r in 0..1, alpha 1


def test_gouraud_native_needs_srgb_and_one_alpha(dummy_bridge, tmp_path):
    svg = _mesh_svg({"mesh_gouraud"}, tmp_path, "auto", color_space="display-p3")
    assert "data-umbruch-mesh" not in svg and "color(display-p3" in svg
    with pytest.raises(ValueError, match="sRGB only"):
        _mesh_svg({"mesh_gouraud"}, tmp_path, "native", color_space="display-p3")
    with pytest.raises(ValueError, match="cannot paint"):
        _mesh_svg(set(), tmp_path, "native")
    # one alpha per vertex that differs: flat triangles in auto
    hb.config.gouraud = "auto"
    hb.config.bridge.capabilities = frozenset({"mesh_gouraud"})
    fig, ax = plt.subplots()
    tri = np.array([[[0, 0], [1, 0], [0, 1]]], dtype=float)
    cols = np.array([[[1, 0, 0, 1], [0, 1, 0, 0.5], [0, 0, 1, 1]]])
    out = io.StringIO()
    renderer = hb.RendererUmbruch(10, 10, out, bridge=hb.config.bridge)
    renderer.draw_gouraud_triangles(renderer.new_gc(), tri, cols, matplotlib.transforms.IdentityTransform())
    renderer.finalize()
    assert "data-umbruch-mesh" not in out.getvalue() and "<path" in out.getvalue()
    plt.close(fig)


def test_single_letter_mathrm_is_upright():
    ml = hb._default_tex_to_mathml(r"\mathrm{s}^{-1}")
    assert 'mathvariant="normal">s<' in ml, ml
    assert 'mathvariant="normal"' not in hb._default_tex_to_mathml("s")


def _children(el):
    return [c.tag.rsplit("}", 1)[-1] + (f"({c.text})" if c.text else "") for c in el]


def test_function_application_stays_outside_scripts():
    root = ET.fromstring(hb._default_tex_to_mathml(r"\sin^2\theta+\cos\theta"))
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] == "msup":
            assert len(el) == 2, _children(el)
    flat = _children(root[0])
    # msup, function application, theta, +, cos, function application, theta
    assert flat[:3] == ["msup", "mo(⁡)", "mi(θ)"], flat
    assert flat[4:6] == ["mi(cos)", "mo(⁡)"], flat
    # a function inside a script that is not its base is still followed by the operator
    ml = hb._default_tex_to_mathml(r"x_{\sin y}")
    assert "⁡" in ml


def test_unknown_macro_raises(dummy_bridge):
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$\definitelynotamacro{x}$")
    with pytest.raises(RuntimeError, match="MathML"):
        fig.savefig(io.BytesIO(), format="pdf")


def test_usetex_raises(dummy_bridge):
    with matplotlib.rc_context({"text.usetex": True}):
        fig, ax = plt.subplots()
        ax.set_title("x")
        with pytest.raises(NotImplementedError):
            fig.savefig(io.BytesIO(), format="pdf")


def test_mathdefault_subset():
    assert hb._mathdefault_html(r"\mathdefault{10^{3}}") == "10<sup>3</sup>"
    with pytest.raises(ValueError):
        hb._mathdefault_html(r"\mathdefault{\alpha}")


def test_font_spec_names_the_file_matplotlib_found():
    from matplotlib import font_manager as fm
    prop = FontProperties(family="DejaVu Serif", style="italic", weight="bold")
    spec = hb._font_spec(prop)
    assert spec["face"]["path"] == fm.findfont(prop)
    assert spec["css"].startswith(f"font-family:'{spec['face']['alias']}', 'DejaVu Serif'")
    assert "font-weight:700;font-style:italic" in spec["css"]
    rules = hb._font_face_css([spec])
    assert rules.count("@font-face") == 1 and "base64," in rules
    assert f"font-family:'{spec['face']['alias']}'" in rules


def test_svg_carries_the_fonts_it_uses(dummy_bridge, tmp_path):
    fig, ax = plt.subplots()
    ax.set_title("x", fontfamily="DejaVu Serif")
    p = tmp_path / "f.svg"
    fig.savefig(p)
    t = p.read_text(encoding="utf-8")
    ET.fromstring(t)
    alias = hb._font_spec(FontProperties(family="DejaVu Serif"))["face"]["alias"]
    assert "@font-face" in t and f"font-family:'{alias}'" in t


@pytest.fixture
def math_font():
    saved = hb.config.math_font
    yield lambda v: setattr(hb.config, "math_font", v)
    hb.config.math_font = saved


def _math(tex=r"$x^2$"):
    return hb.RendererUmbruch(1, 1, io.StringIO(), bridge=Dummy())._markup(tex, True)


def test_math_font_default_and_fontset(math_font):
    assert "style=" not in _math()
    with matplotlib.rc_context({"mathtext.fontset": "cm"}):
        assert "font-family:'Latin Modern Math'" in _math()
    math_font("STIX Two Math")
    assert "font-family:'STIX Two Math'" in _math()


def test_math_font_from_a_file(math_font, tmp_path):
    from matplotlib import font_manager as fm
    path = fm.findfont("DejaVu Sans")      # any font file will do for the markup
    math_font(path)
    alias = hb._math_font()[1]["alias"]
    assert f"font-family:'{alias}'" in _math()
    assert f"font-family:'{alias}'" in hb._font_face_css([], math=True)
    assert hb._font_face_css([], math=False) == ""


def test_unknown_math_font_raises(math_font):
    math_font("No Such Math Font")
    with pytest.raises(ValueError, match="math font"):
        _math()


def test_no_fallback_when_the_server_cannot_start(monkeypatch, tmp_path):
    monkeypatch.setattr(hb.config, "bridge", None)
    monkeypatch.setenv("UMBRUCH", str(tmp_path / "no-such-umbruch"))
    fig, ax = plt.subplots()
    ax.set_title("x")
    with pytest.raises(umbruch.Error, match="cannot start"):
        fig.savefig(io.BytesIO(), format="pdf")
    monkeypatch.delenv("UMBRUCH")
    monkeypatch.setattr(umbruch, "find_umbruch", lambda: None)
    with pytest.raises(umbruch.Error, match="not found"):
        fig.savefig(io.BytesIO(), format="svg")
    plt.close(fig)


def test_private_parts_of_matplotlibs_svg_backend_are_there():
    """What the import-time check cannot see: attributes set by the constructors."""
    from matplotlib.backends.backend_svg import RendererSVG
    writer = hb._XMLWriter(io.StringIO())
    assert callable(writer._XMLWriter__flush) and callable(writer._XMLWriter__write)
    renderer = RendererSVG(10, 10, io.StringIO())
    assert isinstance(renderer._hatchd, dict)
    assert renderer.writer.__class__ is hb._XMLWriter


def test_hatches_in_the_color_space_with_and_without_line_width_in_the_key(dummy_bridge):
    fig, ax = plt.subplots()
    ax.bar([0], [1], hatch="//", facecolor="C1", edgecolor="C2")
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", color_space="display-p3")
    svg = buf.getvalue().decode()
    plt.close(fig)
    assert "<pattern" in svg and "stroke-width" in svg and "color(display-p3" in svg


def test_unknown_gouraud_mode_raises(dummy_bridge):
    hb.config.gouraud = "stock"
    fig = fig_with_everything()
    with pytest.raises(ValueError, match="config.gouraud"):
        fig.savefig(io.BytesIO(), format="svg")


def test_unknown_color_space_raises(dummy_bridge):
    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="unknown color space"):
        fig.savefig(io.BytesIO(), format="svg", color_space="cie-xyz")
    plt.close(fig)


def test_no_hex_colors_are_left_in_a_color_space(dummy_bridge):
    """Everything painted is written in the color space: fills, strokes, text, hatches,
    gradient seams; no 8-bit hex colors survive in the SVG."""
    hb.config.gouraud = "subdivide"
    fig = fig_with_everything()
    ax = fig.axes[0]
    ax.bar([0.2], [0.5], hatch="//", facecolor="C1", edgecolor="C2")
    ax.scatter([0.5], [0.5], c="C3", s=60, marker="s")
    ax.set_title("t", color=(0.2, 0.4, 0.6))
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", color_space="display-p3")
    svg = buf.getvalue().decode()
    plt.close(fig)
    assert "color(display-p3" in svg
    left = re.findall(r"(?:fill|stroke|stop-color|flood-color|color)\s*[:=]\s*[\"']?\s*#[0-9a-fA-F]{6}", svg)
    assert not left, left[:5]
    buf = io.BytesIO()
    fig = fig_with_everything()
    fig.savefig(buf, format="svg", color_space="srgb")     # the default: stock hex
    assert re.search(r"#[0-9a-f]{6}", buf.getvalue().decode())
    plt.close(fig)


def test_painted_colors_keep_full_precision(dummy_bridge, tmp_path):
    from matplotlib.patches import Rectangle
    fig, ax = plt.subplots()
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=(0.1234567, 0.5, 0.9),
                           edgecolor=(0.7654321, 0.2, 0.1), linewidth=2))
    ax.set_title("t", color=(0.3333333, 0.6666667, 0.0123457))
    for space, spaced in (("display-p3", True), ("srgb", False)):
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", color_space=space)
        svg = buf.getvalue().decode()
        assert ("color(display-p3 0.123457 0.5 0.9)" in svg) == spaced, space
        assert ("color(display-p3 0.765432 0.2 0.1)" in svg) == spaced, space
        assert ("color:color(display-p3 0.333333 0.666667 0.0123457)" in svg) == spaced
        assert (matplotlib.colors.rgb2hex((0.1234567, 0.5, 0.9)) in svg) == (not spaced), space
    plt.close(fig)


def test_images_are_written_with_the_profile_of_the_color_space(dummy_bridge):
    import base64
    from PIL import Image
    fig = plt.figure(figsize=(1, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(np.zeros((4, 4, 3)) + (0, 1, 0), interpolation="nearest")
    for space, tagged in (("display-p3", True), ("srgb", False), (None, False)):
        buf = io.BytesIO()
        fig.savefig(buf, format="svg", color_space=space)
        svg = buf.getvalue().decode()
        png = base64.b64decode(re.search(r"data:image/png;base64,\s*([^\"]+)", svg).group(1))
        im = Image.open(io.BytesIO(png))
        assert ("icc_profile" in im.info) == tagged, space
        if tagged:
            assert im.info["icc_profile"] == hb._icc_profile(space)
        assert im.convert("RGB").getpixel((im.size[0] // 2, im.size[1] // 2)) == (0, 255, 0)
    # an image that has a profile already is left alone
    uri = re.search(r"data:image/png;base64,\s*[^\"]+", svg).group(0)
    tagged_uri = hb._tag_png_uri(uri, "display-p3")
    assert hb._tag_png_uri(tagged_uri, "display-p3") == tagged_uri
    plt.close(fig)


@pytest.mark.parametrize("space", ["display-p3", "a98-rgb", "prophoto-rgb", "rec2020"])
def test_icc_profile_agrees_with_convert_color(space):
    """The profile we write and the numbers we convert with describe the same space
    (checked through LittleCMS)."""
    from PIL import Image, ImageCms
    profile = ImageCms.ImageCmsProfile(io.BytesIO(hb._icc_profile(space)))
    to_srgb = ImageCms.buildTransform(profile, ImageCms.createProfile("sRGB"),
                                      "RGB", "RGB", renderingIntent=1)
    for rgb in [(128, 128, 128), (100, 150, 200), (200, 100, 50), (30, 60, 90)]:
        got = ImageCms.applyTransform(Image.new("RGB", (1, 1), rgb), to_srgb).getpixel((0, 0))
        want = np.round(255 * np.array(hb.convert_color(tuple(v / 255 for v in rgb),
                                                          space, "srgb")[:3]))
        assert np.abs(np.array(got) - want).max() <= 1, (space, rgb, got, want)


def test_convert_color_and_cmap():
    assert hb.convert_color("white", "srgb", "display-p3")[:3] == pytest.approx((1, 1, 1), abs=1e-6)
    r, g, b, a = hb.convert_color((0.2, 0.4, 0.6, 0.5), "display-p3", "display-p3")
    assert (r, g, b, a) == (0.2, 0.4, 0.6, 0.5)
    # sRGB red is inside Display P3 and a bit less saturated there
    r, g, b, _ = hb.convert_color("red", "srgb", "display-p3")
    assert 0.9 < r < 1 and 0.1 < g < 0.3
    # Display P3 green is outside sRGB: clipped
    assert hb.convert_color((0, 1, 0), "display-p3", "srgb")[0] == 0
    cm = hb.convert_cmap("viridis", "srgb", "display-p3", N=16)
    assert cm.N == 16 and cm.name == "viridis_display-p3"
    with matplotlib.rc_context():
        hb.config.color_space, saved = "display-p3", hb.config.color_space
        try:
            assert hb.convert_color("red") == hb.convert_color("red", "srgb", "display-p3")
        finally:
            hb.config.color_space = saved


# -- against the real umbruch# -- against the real umbruch ---------------------------------------------------

@need_umbruch
def test_metrics_agree_with_matplotlib(real_bridge):
    assert isinstance(real_bridge, hb.UmbruchServer), type(real_bridge)
    assert isinstance(real_bridge.server, umbruch.Server)
    assert {"measure", "render"} <= real_bridge.capabilities
    prop = FontProperties(size=10)
    for s in ["0.5", "plain label", "Wavy yg"]:
        w, h, d = real_bridge.measure(hb.escape(s), hb._font_spec(prop))
        w0, h0, d0 = TextToPath().get_text_width_height_descent(s, prop, False)
        assert abs(h - h0) < 0.2 and abs(d - d0) < 0.2 and 0.95 * w0 <= w <= 1.2 * w0, \
            (s, w, w0, h, h0, d, d0)
    _, h, _ = real_bridge.measure("E = mc<sup>2</sup>", hb._font_spec(prop))
    _, h0, _ = real_bridge.measure("E = mc2", hb._font_spec(prop))
    assert h > h0, "a superscript must make the box taller"


@need_umbruch
def test_pdf_end_to_end(real_bridge, tmp_path):
    fig = fig_with_everything()
    pdf = tmp_path / "fig.pdf"
    fig.savefig(pdf)
    assert pdf.read_bytes()[:4] == b"%PDF"
    fig.savefig(tmp_path / "tight.pdf", bbox_inches="tight")
    if shutil.which("pdftotext"):
        txt = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True,
                             text=True).stdout
        for word in ("Rate", "plain label", "Hz"):
            assert word in txt, (word, txt)


@need_umbruch
def test_server_restarts_after_a_crash(real_bridge):
    pid = real_bridge.server._proc.pid
    real_bridge.server._proc.kill()
    real_bridge.server._proc.wait()
    assert real_bridge.measure("again", hb._font_spec(FontProperties(size=10)))[0] > 0
    assert real_bridge.server._proc.pid != pid


@need_umbruch
def test_large_batches(real_bridge):
    real_bridge.measure_many([(f"label {i}", hb._font_spec(FontProperties(size=10)))
                              for i in range(3000)])


@need_umbruch
@pytest.mark.skipif(not shutil.which("pdffonts"), reason="needs pdffonts")
def test_tick_label_exponents_use_the_text_font(real_bridge, tmp_path):
    fig, ax = plt.subplots()
    ax.loglog([1, 1e4], [1e-3, 1e3])
    path = tmp_path / "log.pdf"
    fig.savefig(path)
    fonts = subprocess.run(["pdffonts", str(path)], capture_output=True, text=True).stdout
    assert "STIX" not in fonts, fonts


@need_umbruch
@pytest.mark.skipif(not shutil.which("pdffonts"), reason="needs pdffonts")
def test_fonts_not_bundled_with_umbruch_are_used(real_bridge, tmp_path):
    """DejaVu Serif and the italic and bold-italic DejaVu Sans are not among umbruch's
    own fonts; they come from the files Matplotlib found."""
    fig, ax = plt.subplots()
    ax.set_title("serif title", fontfamily="DejaVu Serif")
    ax.set_xlabel("slanted", fontstyle="italic")
    ax.set_ylabel("slanted bold", fontstyle="italic", fontweight="bold")
    path = tmp_path / "fonts.pdf"
    fig.savefig(path)
    fonts = subprocess.run(["pdffonts", str(path)], capture_output=True, text=True).stdout
    assert "DejaVuSerif" in fonts, fonts
    assert "DejaVuSans-Oblique" in fonts, fonts
    assert "DejaVuSans-BoldOblique" in fonts, fonts
    assert "Heros" not in fonts, fonts


@need_umbruch
@pytest.mark.skipif(not shutil.which("pdffonts"), reason="needs pdffonts")
@pytest.mark.parametrize("font, expect", [(None, "STIXTwoMath"),
                                          ("Latin Modern Math", "LatinModernMath")])
def test_math_font_in_the_pdf(real_bridge, math_font, tmp_path, font, expect):
    math_font(font)
    fig, ax = plt.subplots()
    ax.set_title(r"$\int_0^\infty e^{-x^2}dx$")
    path = tmp_path / "m.pdf"
    fig.savefig(path)
    fonts = subprocess.run(["pdffonts", str(path)], capture_output=True, text=True).stdout
    assert expect in fonts, fonts
    assert (font is None) == ("LatinModern" not in fonts), fonts


def _pixel(tmp_path, make, name, space):
    from PIL import Image
    fig = make()
    pdf = tmp_path / f"{name}.pdf"
    fig.savefig(pdf, color_space=space)
    plt.close(fig)
    subprocess.run(["pdftoppm", "-r", "20", "-png", "-singlefile", str(pdf),
                    str(tmp_path / name)], check=True)
    return np.array(Image.open(tmp_path / f"{name}.png").convert("RGB").getpixel((10, 10)))


def _flat(rgb):
    return lambda: plt.figure(figsize=(1, 1), facecolor=rgb)


def _image(rgb):
    def make():
        fig = plt.figure(figsize=(1, 1))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.imshow(np.full((8, 8, 3), rgb), interpolation="nearest", aspect="auto")
        return fig
    return make


@need_umbruch
@pytest.mark.skipif(not shutil.which("pdftoppm"), reason="needs pdftoppm")
@pytest.mark.parametrize("space", ["display-p3", "a98-rgb", "prophoto-rgb", "rec2020"])
def test_flat_colors_are_read_in_the_color_space(real_bridge, tmp_path, space):
    """umbruch's reading of color(<space> ...) agrees with convert_color."""
    rgb = (0.1, 0.9, 0.3)
    got = _pixel(tmp_path, _flat(rgb), "flat", space)
    want = np.round(255 * np.array(hb.convert_color(rgb, space, "srgb")[:3]))
    assert np.abs(got - want).max() <= 6, (space, got, want)
    # and it is not the same numbers read as sRGB
    assert np.abs(got - np.round(255 * np.array(rgb))).max() > 10


@need_umbruch
@pytest.mark.skipif(not shutil.which("pdftoppm"), reason="needs pdftoppm")
@pytest.mark.parametrize("space", ["display-p3", "rec2020"])
def test_images_agree_with_flat_colors_in_the_color_space(real_bridge, tmp_path, space):
    """The same numbers as image pixels and as a flat color end up as the same color.
    umbruch uses the ICC profile of the PNG in the SVG <image> (0.5.2 or later)."""
    rgb = (0.1, 0.9, 0.3)
    a, b = _pixel(tmp_path, _flat(rgb), "flat", space), _pixel(tmp_path, _image(rgb), "img", space)
    assert np.abs(a - b).max() <= 3, (space, a, b)


def _content_streams(pdf):
    out = []
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", pdf, re.S):
        try:
            out.append(zlib.decompress(m.group(1)))
        except zlib.error:
            pass
    return b"".join(out)


@need_umbruch
def test_precise_colors_reach_the_pdf(real_bridge):
    """The PDF has the components as given (0.1234567, which umbruch writes with five
    digits), not rounded to 1/255 steps (0.1216)."""
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(1, 1))
    fig.add_artist(Rectangle((0.1, 0.1), 0.8, 0.8, facecolor=(0.1234567, 0.5, 0.9),
                             transform=fig.transFigure))
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf", color_space="display-p3")
    content = _content_streams(buf.getvalue())
    colors = [[float(x) for x in m.split()]
              for m in re.findall(rb"/CS\d+ cs ([\d. ]+) sc", content)]
    assert any(c == pytest.approx([0.1234567, 0.5, 0.9], abs=1e-4) for c in colors), colors  # umbruch writes 5 digits; 8-bit rounding would be 0.1216


@need_umbruch
@pytest.mark.skipif(not shutil.which("pdftoppm"), reason="needs pdftoppm")
def test_native_mesh_matches_subdivided_triangles_and_is_smaller(real_bridge, tmp_path, monkeypatch):
    from PIL import Image
    if "mesh_gouraud" not in real_bridge.capabilities:
        pytest.skip("this umbruch cannot paint meshes")
    X, Y = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
    results = {}
    for mode in ("native", "subdivide"):
        monkeypatch.setattr(hb.config, "gouraud", mode)
        fig = plt.figure(figsize=(2, 2))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.pcolormesh(X, Y, np.sin(5 * X) * np.cos(5 * Y), shading="gouraud", cmap="viridis")
        pdf = tmp_path / f"{mode}.pdf"
        fig.savefig(pdf)
        plt.close(fig)
        subprocess.run(["pdftoppm", "-r", "40", "-png", "-singlefile", str(pdf),
                        str(tmp_path / mode)], check=True)
        results[mode] = (pdf.stat().st_size, np.asarray(Image.open(tmp_path / f"{mode}.png").convert("RGB"), dtype=int))
    assert np.abs(results["native"][1] - results["subdivide"][1]).mean() < 4
    assert results["native"][0] < results["subdivide"][0] / 2
