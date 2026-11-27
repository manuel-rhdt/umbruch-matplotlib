"""Tests; those that need the real umbruch binary ($UMBRUCH, the package or PATH) are
skipped without it.  Run with ``pytest tests``."""
import io
import os
import re
import shutil
import subprocess
import sys
import warnings
import zlib
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest
import matplotlib
matplotlib.use("module://umbruch_matplotlib")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
import umbruch_matplotlib as hb

need_umbruch = pytest.mark.skipif(hb.find_umbruch() is None, reason="umbruch not found")


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


@pytest.mark.parametrize("mode", ["subdivide", "native", "stock"])
def test_gouraud_modes(dummy_bridge, tmp_path, mode):
    hb.config.gouraud = mode
    fig = fig_with_everything()
    p = tmp_path / f"{mode}.svg"
    fig.savefig(p)
    t = p.read_text(encoding="utf-8")
    ET.fromstring(t)
    assert ("data-colors" in t) == (mode == "native")
    assert ("colorAdd" in t) == (mode == "stock")
    assert "foreignObject" in t and "<i>Rate</i> r<sub>0</sub>" in t


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
    with pytest.raises(OSError):
        fig.savefig(io.BytesIO(), format="pdf")
    monkeypatch.setattr(hb, "find_umbruch", lambda: None)
    with pytest.raises(RuntimeError, match="not found"):
        fig.savefig(io.BytesIO(), format="svg")
    plt.close(fig)


def test_wide_colors_replace_painted_colors_only():
    svg = ('<path fill="#FF0000" stroke="#00ff00"/><g style="color: #ff0000;fill:#0000ff"/>'
           '<stop stop-color="#ff0000"/><div>#ff0000 in a label</div>')
    out = hb._apply_wide_colors(svg, {"#ff0000": "color(rec2020 1 0 0)"})
    assert out.count("color(rec2020 1 0 0)") == 3, out
    assert 'stroke="#00ff00"' in out and "fill:#0000ff" in out
    assert "<div>#ff0000 in a label</div>" in out
    assert hb._apply_wide_colors(svg, {}) == svg


# -- against the real umbruch ---------------------------------------------------

@need_umbruch
def test_metrics_agree_with_matplotlib(real_bridge):
    assert isinstance(real_bridge, hb.UmbruchServer), type(real_bridge)
    assert real_bridge.request("hello", {})[0]["protocol"] == 1
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
    pid = real_bridge._proc.pid
    real_bridge._proc.kill()
    real_bridge._proc.wait()
    assert real_bridge.measure("again", hb._font_spec(FontProperties(size=10)))[0] > 0
    assert real_bridge._proc.pid != pid


@need_umbruch
def test_large_batches(real_bridge):
    real_bridge.measure_many([(f"label {i}", hb._font_spec(FontProperties(size=10)))
                              for i in range(3000)])


@need_umbruch
@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_forked_child_gets_its_own_server(real_bridge):
    real_bridge.measure("parent", hb._font_spec(FontProperties(size=10)))
    pid = os.fork()
    if pid == 0:
        try:
            ok = hb.get_bridge().measure("child", hb._font_spec(FontProperties(size=10)))[0] > 0
            os._exit(0 if ok else 1)
        except BaseException:
            os._exit(2)
    _, status = os.waitpid(pid, 0)
    assert status == 0
    assert real_bridge.measure("parent again", hb._font_spec(FontProperties(size=10)))[0] > 0


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


@need_umbruch
def test_wide_colors_reach_the_pdf(real_bridge, monkeypatch):
    monkeypatch.setattr(hb.config, "wide_colors", {"#123456": "color(rec2020 1 0 0)"})
    fig, ax = plt.subplots()
    ax.bar([0], [1], color="#123456")
    plain, wide = io.BytesIO(), io.BytesIO()
    fig.savefig(wide, format="pdf")
    monkeypatch.setattr(hb.config, "wide_colors", {})
    fig.savefig(plain, format="pdf")
    # a wide color is painted in an ICC space: "/CS1 cs <components> sc"
    def painted_in_icc_space(pdf):
        return any(re.search(rb"/CS\d+ cs [\d. ]+ sc", zlib.decompress(m.group(1)))
                   for m in re.finditer(rb"stream\r?\n(.*?)endstream", pdf, re.S)
                   if _inflates(m.group(1)))
    assert painted_in_icc_space(wide.getvalue())
    assert not painted_in_icc_space(plain.getvalue())


def _inflates(data):
    try:
        zlib.decompress(data)
    except zlib.error:
        return False
    return True
