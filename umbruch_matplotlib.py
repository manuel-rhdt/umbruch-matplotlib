"""Matplotlib backend that hands text layout and PDF writing to umbruch.

Pipeline
--------
Matplotlib draws into a subclass of its SVG renderer, which already handles
paths, markers, clipping, hatches, images and collections.  Three things
differ from stock SVG output:

* Text (plain, rich, and math) is written as an XHTML/MathML fragment inside a
  <foreignObject>, which umbruch lays out and draws (it supports HTML and MathML
  there).  Measuring goes through umbruch too, so Matplotlib's layout (tick
  labels, legends, constrained/tight layout, bbox_inches="tight") sees the
  widths umbruch will produce.  Heights and descents are the ink extents of the
  plain text, as with stock Matplotlib, plus what markup adds to the line box
  (sub/superscripts, fractions); the line box itself is used only to place the
  text on its baseline.
* Gouraud triangles (``config.gouraud == "native"``, the default) are one compact
  ``<g data-umbruch-mesh="gouraud" data-vertices=... data-triangles=...>`` element, which
  umbruch paints as a single type 4 shading.  This is an umbruch extension of SVG that no
  other renderer draws, and it needs one alpha per mesh.  ``"subdivide"`` (flat
  sub-triangles) and ``"stock"`` (Matplotlib's gradient emulation) are polyfills for
  other renderers.  None works in a wide color space yet: that raises.
* Fonts: the file that Matplotlib's font manager picks for each text is sent to
  umbruch as an @font-face with a data: URI (in the SVG, and in measure requests),
  under a name derived from the file; the font's own family list follows as fallback.
  The font of math (``config.math_font``, or ``mathtext.fontset``) is set by a style
  attribute on each <math>, since a style sheet would not reach the probes of a measure.
* The finished SVG is rendered to PDF by ``umbruch figure.svg figure.pdf``.

Usage
-----
    import matplotlib
    matplotlib.use("module://umbruch_matplotlib")     # or savefig(..., backend=...)
    import umbruch_matplotlib as hb
    ax.set_title(hb.rich("<i>Rate</i> r<sub>0</sub> in <b>Hz</b>"))
    ax.set_xlabel(r"$\alpha^2$ (needs the latex2mathml package)")
    fig.savefig("out.pdf")

The ``umbruch`` executable is taken from $UMBRUCH, the ``umbruch`` package or PATH.  Rich text is marked by
a leading U+FEFF (see ``rich``), which survives Matplotlib's string handling.
The markup must be well-formed XML (numeric entities, not &nbsp;), and a newline
splits lines as for any Matplotlib text.  ``$...$`` in a plain or rich string is
converted to MathML by ``config.tex_to_mathml`` (default: latex2mathml).  Math
that cannot be converted raises, as does ``text.usetex``; there is no fallback to
Matplotlib's own text rendering.

Tested against Matplotlib 3.11 and umbruch 0.5; it uses a few private attributes of
the SVG renderer.
"""
import base64
import functools
import hashlib
import io
import itertools
import os
import re
import logging
import struct
import zlib
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np

import umbruch

import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends import backend_svg as _backend_svg
from matplotlib.backends.backend_svg import FigureCanvasSVG, RendererSVG

# This backend builds on parts of Matplotlib's SVG backend that are not public API.
# Before 3.11 the XML writer is called XMLWriter; check the rest, so that a Matplotlib
# that changed them fails here with a message rather than somewhere in a drawing.
_XMLWriter = getattr(_backend_svg, "_XMLWriter", None) or _backend_svg.XMLWriter
_generate_css = getattr(_backend_svg, "_generate_css", None)
_PRIVATE = {
    "RendererSVG": [(RendererSVG, n) for n in (
        "_get_clip_attrs", "_make_flip_transform", "_convert_path", "_get_style_dict",
        "_write_hatches", "_get_hatch", "_make_id", "draw_gouraud_triangles")],
    "the XML writer": [(_XMLWriter, n) for n in (
        "_XMLWriter__flush", "start", "end", "element", "data")],
    "backend_svg": [(_backend_svg, "_generate_css")],
}
_missing = [f"{where}.{name}" for where, items in _PRIVATE.items()
            for obj, name in items if not hasattr(obj, name)]
if _missing:
    raise ImportError(
        f"umbruch_matplotlib needs parts of Matplotlib's SVG backend that Matplotlib "
        f"{mpl.__version__} does not have ({', '.join(_missing)}); it is tested with "
        "Matplotlib 3.9 to 3.11")
from matplotlib.colors import rgb2hex
from matplotlib.transforms import Affine2D

log = logging.getLogger(__name__)

XHTML = "http://www.w3.org/1999/xhtml"
RICH_MARK = "﻿"

FigureManager = FigureManagerBase


def rich(markup):
    """Mark *markup* (an XHTML fragment, optionally containing $math$) as rich text."""
    return RICH_MARK + markup


def _fmt(v):
    return f"{v:.3f}".rstrip("0").rstrip(".") or "0"


_ARITY = {"mfrac": 2, "mroot": 2, "msub": 2, "msup": 2, "munder": 2, "mover": 2,
          "msubsup": 3, "munderover": 3}
_MATHML_NS = "http://www.w3.org/1998/Math/MathML"
ET.register_namespace("", _MATHML_NS)
#: operators TeX sets as functions: upright name, thin space, then the argument
_FUNCTIONS = {"sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan",
              "sinh", "cosh", "tanh", "coth", "log", "ln", "lg", "exp", "det", "dim",
              "ker", "deg", "arg", "gcd", "hom", "Pr", "sgn"}
#: mathematical symbols that stay identifiers (everything else in Sm is an operator)
_NOT_OPERATORS = {"\u2202", "\u2207", "\u221e"}


def _fix_mathml(ml):
    """Validate and repair what latex2mathml writes.

    It is lenient: an unknown macro becomes the literal text ``\\name`` and a malformed
    ``\\frac`` becomes an <mfrac> with one child; both raise ValueError.  It also marks
    up things MathML needs marked for correct spacing: binary and relation symbols such
    as \\pm come out as <mi> (no operator spacing), function names as \\sin have no
    function application (no space before the argument), and the periods in
    ``\\mathrm{a.u.}`` are operators (spaced).  These are repaired.
    """
    import re
    import unicodedata
    # latex2mathml drops the upright style of a single-letter \mathrm{s}; it is the only
    # input that yields the zero-padded ASCII entity, so mark exactly those as upright.
    ml = re.sub(r"<mi>(&#x000[0-9A-Fa-f]{2};)</mi>", r'<mi mathvariant="normal">\1</mi>', ml)
    ns = "{%s}" % _MATHML_NS
    root = ET.fromstring(ml)
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if el.text and "\\" in el.text:
            raise ValueError(f"unknown TeX command in {el.text.strip()!r}")
        if tag in _ARITY and len(el) != _ARITY[tag]:
            raise ValueError(f"<{tag}> needs {_ARITY[tag]} arguments, got {len(el)}")
        txt = el.text or ""
        if tag == "mi" and len(txt) == 1 and unicodedata.category(txt) == "Sm" \
                and txt not in _NOT_OPERATORS:
            el.tag = ns + "mo"
        elif tag == "mo" and txt == ".":
            el.tag = ns + "mi"
    # Function application goes after the function, which for \sin^2 means after the
    # whole <msup>: inside it, the script would get three children.
    parents = {child: parent for parent in root.iter() for child in parent}
    local = lambda e: e.tag.rsplit("}", 1)[-1]
    funcs = [el for el in root.iter() if local(el) == "mi" and (el.text or "") in _FUNCTIONS]
    for el in reversed(funcs):
        target = el
        while target in parents and local(parents[target]) in _ARITY \
                and parents[target][0] is target:
            target = parents[target]
        parent = parents.get(target)
        if parent is None:
            continue
        i = list(parent).index(target)
        if i + 1 < len(parent) and local(parent[i + 1]) != "mo":
            fa = ET.Element(ns + "mo")
            fa.text = "\u2061"
            parent.insert(i + 1, fa)
    return ET.tostring(root, encoding="unicode")


def _default_tex_to_mathml(tex):
    try:
        from latex2mathml.converter import convert
    except ImportError:
        raise RuntimeError("math needs the latex2mathml package (pip install latex2mathml) "
                           "or config.tex_to_mathml") from None
    return _fix_mathml(convert(tex))


class _Config:
    #: UmbruchBridge instance; if None, one is created for the umbruch executable
    #: found by ``umbruch.Server``.
    bridge = None
    #: callable(tex: str) -> MathML string, or None to use latex2mathml (an error
    #: if that is not installed).
    tex_to_mathml = None
    #: "native" (default): the mesh is one compact ``<g data-umbruch-mesh="gouraud">``
    #: element that umbruch writes as a single type 4 shading.  This is an umbruch
    #: extension of SVG and nothing else draws it; it needs an umbruch that reports
    #: the mesh_gouraud capability, and one alpha for the whole mesh (otherwise it
    #: raises).  The alternatives are polyfills for other renderers: "subdivide": flat
    #: sub-triangles (works with any renderer); "stock": Matplotlib's three-gradient
    #: filter emulation; "auto": "native" if that is possible, else "subdivide".  None
    #: of them works in a wide ``color_space`` yet (umbruch paints meshes in sRGB).
    gouraud = "native"
    gouraud_max_level = 8
    #: Font of ``$...$`` math: a family name (umbruch's own "STIX Two Math" and "Latin
    #: Modern Math", or any OpenType math font Matplotlib finds) or the path of a font
    #: file.  None follows ``mathtext.fontset``: "cm" is Latin Modern Math, anything
    #: else STIX Two Math.
    math_font = None
    #: The color space of the numbers Matplotlib works with, one of "srgb" (the
    #: default), "display-p3", "a98-rgb", "prophoto-rgb" and "rec2020".  Matplotlib
    #: has no color management, so every RGB value of a figure (colors, colormaps,
    #: image pixels) is just three numbers in [0, 1]; this says which space they are
    #: in.  Painted colors are written as CSS ``color(<space> r g b)`` and raster
    #: images get a matching ICC profile.  ``savefig(..., color_space=...)`` overrides
    #: it for one file.  Colors given as sRGB (hex, names, the default cycle and
    #: colormaps) are only right if converted: see ``convert_color``.
    color_space = None


config = _Config()


# -- color spaces ----------------------------------------------------------------------

_D50 = np.array([0.9642, 1.0, 0.8249])
_D65_XY = (0.3127, 0.3290)
#: primaries (xy), white point (xy) and transfer function ("srgb" or a gamma) of the
#: predefined spaces of CSS Color 4 (rec2020 as umbruch sets it: a gamma of 2.4)
_SPACES = {
    "srgb": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060), _D65_XY, "srgb"),
    "display-p3": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), _D65_XY, "srgb"),
    "a98-rgb": ((0.640, 0.330), (0.210, 0.710), (0.150, 0.060), _D65_XY, 563 / 256),
    "prophoto-rgb": ((0.7347, 0.2653), (0.1596, 0.8404), (0.0366, 0.0001),
                     (0.3457, 0.3585), 1.8),
    "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), _D65_XY, 2.4),
}
_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367],
                      [0.0389, -0.0685, 1.0296]])


def _space(name):
    try:
        return _SPACES[name]
    except KeyError:
        raise ValueError(f"unknown color space {name!r}: one of "
                         f"{', '.join(_SPACES)}") from None


def _xy_to_xyz(xy):
    x, y = xy
    return np.array([x / y, 1.0, (1 - x - y) / y])


@functools.lru_cache(maxsize=None)
def _to_xyz_d50(name):
    """Matrix from linear RGB of *name* to XYZ with a D50 white (the ICC connection space)."""
    *primaries, white_xy, _ = _space(name)
    P = np.array([_xy_to_xyz(p) for p in primaries]).T
    white = _xy_to_xyz(white_xy)
    M = P * np.linalg.solve(P, white)
    gain = np.diag((_BRADFORD @ _D50) / (_BRADFORD @ white))
    return np.linalg.inv(_BRADFORD) @ gain @ _BRADFORD @ M


def _decode(v, trc):
    v = np.asarray(v, dtype=float)
    a = np.abs(v)
    lin = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4) \
        if trc == "srgb" else a ** trc
    return np.sign(v) * lin


def _encode(v, trc):
    v = np.asarray(v, dtype=float)
    a = np.abs(v)
    enc = np.where(a <= 0.0031308, a * 12.92, 1.055 * a ** (1 / 2.4) - 0.055) \
        if trc == "srgb" else a ** (1 / trc)
    return np.sign(v) * enc


def convert_color(color, src="srgb", dst=None):
    """*color* (any Matplotlib color, or an array of RGB(A) rows) given in the color
    space *src*, as RGB(A) numbers in *dst* (default: ``config.color_space``), clipped
    to the gamut of *dst*.  Use it for the colors that Matplotlib supplies as sRGB
    (names, hex, the default colors) when the figure is in another space."""
    dst = dst or config.color_space or "srgb"
    single = isinstance(color, str) or (np.ndim(color) == 1 and len(color) in (3, 4)
                                        and not isinstance(color[0], str))
    rgba = mpl.colors.to_rgba_array(color)
    if src != dst:
        lin = _decode(rgba[:, :3], _space(src)[4]) @ _to_xyz_d50(src).T
        out = _encode(np.linalg.solve(_to_xyz_d50(dst), lin.T).T, _space(dst)[4])
        rgba = np.concatenate([np.clip(out, 0, 1), rgba[:, 3:]], axis=1)
    return tuple(rgba[0]) if single else rgba


def convert_cmap(cmap, src="srgb", dst=None, N=256):
    """A colormap (a name or a Colormap) whose colors are given in *src*, as a
    ListedColormap of *N* colors in *dst* (see ``convert_color``)."""
    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    colors = convert_color(cmap(np.linspace(0, 1, N)), src, dst)
    return mpl.colors.ListedColormap(colors, name=f"{cmap.name}_{dst or config.color_space}")


def _icc_profile(name):
    """A matrix/TRC ICC profile (version 2) of the color space *name*."""
    *_, trc = _space(name)
    s15 = lambda v: struct.pack(">i", int(round(v * 65536)))
    xyz = lambda v: b"XYZ \0\0\0\0" + b"".join(s15(c) for c in v)
    x = np.linspace(0, 1, 1024)
    curve = b"curv\0\0\0\0" + struct.pack(">I", len(x)) + np.round(
        _decode(x, trc) * 65535).astype(">u2").tobytes()
    text = name.encode("ascii") + b"\0"
    desc = (b"desc\0\0\0\0" + struct.pack(">I", len(text)) + text + b"\0" * 8 + b"\0\0\0"
            + b"\0" * 67)
    M = _to_xyz_d50(name)
    tags = {b"desc": desc, b"cprt": b"text\0\0\0\0" + b"public domain\0",
            b"wtpt": xyz(_D50), b"rXYZ": xyz(M[:, 0]), b"gXYZ": xyz(M[:, 1]),
            b"bXYZ": xyz(M[:, 2]), b"rTRC": curve, b"gTRC": curve, b"bTRC": curve}
    offset = 128 + 4 + 12 * len(tags)
    table, body = b"", b""
    for sig, data in tags.items():
        data += b"\0" * (-len(data) % 4)
        table += sig + struct.pack(">II", offset + len(body), len(data))
        body += data
    size = offset + len(body)
    header = (struct.pack(">I", size) + b"\0\0\0\0" + b"\x02\x40\x00\x00" + b"mntrRGB XYZ "
              + b"\x07\xea\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00" + b"acsp" + b"\0" * 24
              + b"\0\0\0\0" + b"".join(s15(c) for c in _D50) + b"\0" * 4 + b"\0" * 16
              + b"\0" * 28)
    return header + struct.pack(">I", len(tags)) + table + body


def _png_chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(
        ">I", zlib.crc32(kind + data))


def _tag_png_uri(uri, space):
    """The ``data:image/png;base64,...`` URI *uri* with the ICC profile of *space* in
    the PNG (after the IHDR chunk), unless it has one already."""
    head, _, data = uri.partition("base64,")
    png = base64.b64decode(data)
    end = 8 + 12 + 13        # the signature and the IHDR chunk
    if png[:8] != b"\x89PNG\r\n\x1a\n" or b"iCCP" in png[:end + 64]:
        return uri
    iccp = _png_chunk(b"iCCP", space.encode() + b"\0\0" + zlib.compress(_icc_profile(space)))
    return head + "base64," + base64.b64encode(png[:end] + iccp + png[end:]).decode("ascii")


class UmbruchBridge:
    """What the backend needs from a renderer.  Units are SVG user units, which are
    CSS px inside a foreignObject and, in Matplotlib's SVG, points."""

    def __init__(self):
        self._cache = {}     # key -> (w, h, d) as Matplotlib should see it
        self._ascent = {}    # key -> distance from the top of the line box to the baseline
        self._collect = None
        #: what the renderer can do beyond the basics, e.g. {"mesh_gouraud"}
        self.capabilities = frozenset()

    # to implement -----------------------------------------------------------
    def measure_many(self, items):
        """Measure [(markup, font), ...]; must fill ``self._cache`` for every item."""
        raise NotImplementedError

    def render(self, svg, width, height):
        """Return PDF bytes for the intermediate document."""
        raise NotImplementedError

    # provided ---------------------------------------------------------------
    @staticmethod
    def key(markup, font):
        return (markup, font["css"])

    def measure(self, markup, font):
        """(width, height, descent) of *markup* set in *font*; the first baseline is
        ``height - descent`` below the top."""
        if markup == "":
            return (0.0, 0.0, 0.0)
        key = self.key(markup, font)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if self._collect is not None:
            # a prefetch pass: remember what is missing, answer roughly
            self._collect[key] = (markup, font)
            n = len(_TAG_RE.sub("", markup))
            return (0.6 * font["size"] * n, 1.4 * font["size"], 0.35 * font["size"])
        self.measure_many([(markup, font)])
        return self._cache[key]

    def ascent(self, markup, font):
        """(ascent, height) of the renderer's own line box: this is where the text is
        placed, not what Matplotlib is told."""
        key = self.key(markup, font)
        if key not in self._ascent:
            self.measure(markup, font)
        return self._ascent.get(key, (1.1 * font["size"], 1.4 * font["size"]))

    def close(self):
        pass


class UmbruchServer(UmbruchBridge):
    """The bridge to umbruch: an ``umbruch.Server`` (one long-lived ``umbruch --serve``
    process; the executable is ``$UMBRUCH``, the one installed with the package or the
    one on PATH).  It sees only the fonts built into umbruch, so that the output does
    not depend on the machine: the fonts of the text are sent along as @font-face.
    ``umbruch.Error`` is raised if it cannot be found or started."""

    BATCH = 2000

    def __init__(self, server=None):
        super().__init__()
        self.server = server or umbruch.Server(system_fonts=False)
        self.server.measure([])          # starts the process now, so that errors surface here
        self.capabilities = self.server.capabilities
        missing = {"measure", "render"} - self.capabilities
        if missing:
            raise umbruch.Error("server", f"umbruch --serve lacks {sorted(missing)}")

    _requests = itertools.count()

    @classmethod
    def _fresh_aliases(cls, rules, probes):
        """*rules* and *probes* [(markup, css), ...] with the family name of every embedded face made
        unique to this request.  umbruch's server (0.5.2) applies an @font-face family only in the
        first request that declares it: a later request that declares it again, or none, measures
        in the fallback font (for Noto Sans 9 % too narrow, so that text was cut off where
        it overflowed its box).  The names in the cache keys and in the SVG stay as they are."""
        tag = f"-{next(cls._requests)}"
        aliases = set(re.findall(r"font-family:'(mpl-[0-9a-f]+)'", rules))

        def rename(text):
            for alias in aliases:
                text = text.replace(alias, alias + tag)
            return text
        return rename(rules), [(rename(m), rename(css)) for m, css in probes]

    def measure_many(self, items):
        items = [it for it in dict((self.key(m, f), (m, f)) for m, f in items).values()
                 if self.key(*it) not in self._cache]
        for start in range(0, len(items), self.BATCH):
            chunk = items[start:start + self.BATCH]
            # @font-face rules apply to the probes of the same request
            rules = _font_face_css((f for _, f in chunk),
                                   math=any("<math" in m for m, _ in chunk))
            probes = [(m, f["css"]) for m, f in chunk]
            if rules:
                rules, probes = self._fresh_aliases(rules, probes)
            head = [{"markup": f"<style>{rules}</style>", "css": ""}] if rules else []
            metrics = self.server.measure(head + [
                {"markup": m, "css": css} for m, css in probes])
            for (markup, font), m in zip(chunk, metrics[len(head):]):
                if "error" in m:
                    raise RuntimeError(f"umbruch could not measure {markup!r}: {m['error']}")
                ink = m["ink"]
                if ink is None:  # nothing painted (spaces)
                    h = d = 0.0
                else:
                    d = max(ink["bottom"], 0.0)
                    h = ink["bottom"] - ink["top"]
                key = self.key(markup, font)
                self._cache[key] = (m["width"], h, d)
                self._ascent[key] = (m["ascent"], m["height"])
                if m.get("missing_glyphs"):
                    log.warning("no font has %s (in %r)", ", ".join(m["missing_glyphs"]),
                                markup[:40])

    def render(self, svg, width, height):
        rendered = umbruch.HTML(svg, xml=True, server=self.server).render()
        for w in rendered.warnings:
            if "PDF/UA" not in str(w):  # not a goal of a plot
                log.warning("umbruch: %s", w)
        return rendered.pdf

    def close(self):
        self.server.close()


def get_bridge():
    """The bridge to umbruch (started on first use); raises ``umbruch.Error`` if umbruch
    cannot be found or started.  There is no fallback to another way of running it."""
    if config.bridge is None:
        config.bridge = UmbruchServer()
    return config.bridge


@functools.lru_cache(maxsize=None)
def _font_data_uri(path):
    with open(path, "rb") as fh:
        data = base64.b64encode(fh.read()).decode("ascii")
    return f"data:{'font/otf' if path.lower().endswith('.otf') else 'font/ttf'};base64,{data}"


@functools.lru_cache(maxsize=None)
def _face_of(path):
    """The face Matplotlib found at *path*, as a description for @font-face."""
    entry = next((e for e in fm.fontManager.ttflist if e.fname == path), None)
    weight = int(fm.weight_dict.get(entry.weight, entry.weight)) if entry else None
    ident = f"{path}:{os.path.getmtime(path)}"
    return {"alias": "mpl-" + hashlib.sha1(ident.encode()).hexdigest()[:12], "path": path,
            "weight": weight, "style": entry.style if entry else None}


def _face_css(faces):
    return "".join(
        f"@font-face{{font-family:'{f['alias']}';font-weight:{f['weight']};"
        f"font-style:{f['style']};src:url({_font_data_uri(f['path'])})}}"
        for f in {f["alias"]: f for f in faces}.values())


def _font_face_css(fonts, math=False):
    """@font-face rules (the font files as data: URIs) for the fonts of *fonts*, and for
    the math font if *math* and it is a file."""
    faces = [f["face"] for f in fonts if f.get("face")]
    if math:
        faces.append(_math_font()[1])
    return _face_css(f for f in faces if f)


_MATH_BUNDLED = {"stix two math", "latin modern math"}


def _math_font():
    """(CSS font-family value, face to embed or None) of the math font."""
    value = config.math_font
    if value is None:
        value = "Latin Modern Math" if mpl.rcParams["mathtext.fontset"] == "cm" else None
    return _math_font_for(value)


@functools.lru_cache(maxsize=None)
def _math_font_for(value):
    if value is None:
        return None, None
    if value.lower() in _MATH_BUNDLED:
        return repr(value), None
    path = value if os.path.isfile(value) else None
    if path is None:
        try:
            path = fm.findfont(fm.FontProperties(family=value), fallback_to_default=False)
        except ValueError:
            raise ValueError(f"math font {value!r} not found: give the family of an "
                             "OpenType math font Matplotlib can find, or a file") from None
    ident = f"{path}:{os.path.getmtime(path)}"
    face = {"alias": "mpl-math-" + hashlib.sha1(ident.encode()).hexdigest()[:12],
            "path": path, "weight": 400, "style": "normal"}
    return repr(face["alias"]), face


def _style_math(ml):
    """Set the math font on the <math> elements of *ml* (a rule in a style sheet would
    not reach the probes of a measure request, a style attribute does)."""
    family = _math_font()[0]
    if family is None:
        return ml
    return re.sub(r"<math\b", f'<math style="font-family:{family}"', ml)


def _font_spec(prop):
    """What the renderer needs to set text in *prop*: the CSS, and the font file that
    Matplotlib itself would use, which is sent along as an @font-face (umbruch cannot
    see the system's fonts, and this way it sets exactly the face Matplotlib measured
    for).  The families of *prop* follow as fallbacks."""
    weight = prop.get_weight()
    weight = fm.weight_dict.get(weight, weight)
    try:
        face = _face_of(fm.findfont(prop))
    except (ValueError, OSError):
        face = None
    if face and face["weight"] is not None:
        weight, style = face["weight"], face["style"]
    else:
        face, style = None, prop.get_style()
    names = []
    for entry in prop.get_family():
        entry = "sans-serif" if entry in ("sans", "sans serif") else entry
        if entry in fm.font_family_aliases:
            names += ["sans-serif" if n in ("sans", "sans serif") else n
                      for n in fm.FontManager._expand_aliases(entry)]
        names.append(entry)
    names = list(dict.fromkeys(names))
    quoted = [n if n in fm.font_family_aliases or n == "sans-serif" else repr(n)
              for n in names]
    if face:
        quoted.insert(0, repr(face["alias"]))
    return {"prop": prop, "family": names, "size": float(prop.get_size()), "weight": int(weight),
            "face": face, "style": style, "variant": prop.get_variant(),
            "stretch": prop.get_stretch(),
            "css": (f"font-family:{', '.join(quoted)};font-size:{_fmt(prop.get_size())}px;"
                    f"font-weight:{int(weight)};font-style:{style};"
                    f"font-variant:{prop.get_variant()}"
                    + ("" if prop.get_stretch() == "normal" else
                       f";font-stretch:{prop.get_stretch()}"))}


_TEXT_MACROS = {"times": "\u00d7", "cdot": "\u00b7", "pm": "\u00b1", ",": "\u2009",
                ";": "\u2005", " ": " ", "%": "%", "_": "_", "{": "{", "}": "}",
                "$": "$", "#": "#", "&": "&"}


def _mathdefault_html(tex):
    """Matplotlib wraps tick labels in ``\\mathdefault{...}`` to say "set this in the
    regular text font" (``10^{3}``, ``1\\times10^{-2}``).  That is text with scripts,
    not math, so it becomes inline XHTML in the surrounding font.  Anything beyond that
    small subset raises ValueError."""
    def group(i, closing):
        out = []
        while i < len(tex):
            c = tex[i]
            if c == "}":
                if closing:
                    return "".join(out), i + 1
                raise ValueError(f"unbalanced '}}' in {tex!r}")
            if c == "\\":
                j = i + 1
                if j < len(tex) and tex[j].isalpha():
                    while j < len(tex) and tex[j].isalpha():
                        j += 1
                    name = tex[i + 1:j]
                else:
                    name, j = tex[i + 1:j + 1], j + 1
                if name == "mathdefault":
                    if j >= len(tex) or tex[j] != "{":
                        raise ValueError(f"\\mathdefault needs a braced argument in {tex!r}")
                    inner, i = group(j + 1, True)
                    out.append(inner)
                elif name in _TEXT_MACROS:
                    out.append(escape(_TEXT_MACROS[name]))
                    i = j
                else:
                    raise ValueError(f"\\{name} is not supported in \\mathdefault text "
                                     f"{tex!r}")
            elif c in "^_":
                tag = "sup" if c == "^" else "sub"
                i += 1
                if i >= len(tex):
                    raise ValueError(f"dangling {c!r} in {tex!r}")
                if tex[i] == "{":
                    inner, i = group(i + 1, True)
                else:
                    inner, i = group_one(i)
                out.append(f"<{tag}>{inner}</{tag}>")
            elif c == "{":
                inner, i = group(i + 1, True)
                out.append(inner)
            else:
                out.append(escape("\u2212" if c == "-" else c))
                i += 1
        if closing:
            raise ValueError(f"unbalanced '{{' in {tex!r}")
        return "".join(out), i

    def group_one(i):
        # a single character or macro as script argument
        if tex[i] == "\\":
            j = i + 2 if not tex[i + 1:i + 2].isalpha() else i + 1
            while j < len(tex) and tex[i + 1:i + 2].isalpha() and tex[j].isalpha():
                j += 1
            return group_sub(tex[i:j]), j
        return escape("\u2212" if tex[i] == "-" else tex[i]), i + 1

    def group_sub(piece):
        return _mathdefault_html(piece)

    return group(0, False)[0]


_MATH_RE = re.compile(r"(?<!\\)\$(.+?)(?<!\\)\$", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


_MATH_EL_RE = re.compile(r"<math\b.*?</math>", re.S)


def _plain_text(markup):
    """The text of a markup fragment, without tags and without math (as XML text)."""
    import html
    text = html.unescape(_TAG_RE.sub("", _MATH_EL_RE.sub("", markup)))
    return text if text.strip() else "x"


def _ink_height_descent(text, font):
    from matplotlib.textpath import TextToPath
    _, h, d = TextToPath().get_text_width_height_descent(text, font["prop"], False)
    return h, d


def _check_xml(fragment):
    try:
        ET.fromstring(f'<div xmlns="{XHTML}">{fragment}</div>')
    except ET.ParseError as e:
        raise ValueError(f"rich text is not well-formed XML ({e}): {fragment!r}") from None


class _RawWriter(_XMLWriter):
    #: the color space of the figure: PNG images are written with its ICC profile
    color_space = None

    def element(self, tag, text="", attrib={}, **extra):
        href = attrib.get("xlink:href", "")
        if tag == "image" and self.color_space not in (None, "srgb") \
                and href.startswith("data:image/png;base64,"):
            attrib = {**attrib, "xlink:href": _tag_png_uri(href, self.color_space)}
        super().element(tag, text, attrib, **extra)

    def raw(self, s):
        # Pretty-printing would insert whitespace between inline elements.
        self._XMLWriter__flush()
        self._XMLWriter__write(s)


class RendererUmbruch(RendererSVG):
    def __init__(self, width, height, svgwriter, basename=None, image_dpi=72,
                 *, metadata=None, bridge, color_space=None):
        super().__init__(width, height, svgwriter, basename, image_dpi,
                         metadata=metadata)
        self.writer.__class__ = _RawWriter
        self.bridge = bridge
        self.color_space = color_space
        self.writer.color_space = color_space
        self._fonts = {}
        self._math = False

    # -- colors -------------------------------------------------------------

    def _css_color(self, rgb):
        """*rgb* as a CSS color: in the figure's color space at full precision (stock
        SVG writes hex, which rounds to 1/255), or as hex for sRGB."""
        if self.color_space in (None, "srgb"):
            return rgb2hex(rgb)
        return "color({} {})".format(self.color_space,
                                     " ".join(f"{float(v):.6g}" for v in rgb[:3]))

    def _get_style_dict(self, gc, rgbFace):
        attrib = super()._get_style_dict(gc, rgbFace)
        if self.color_space not in (None, "srgb"):
            if attrib.get("fill", "").startswith("#"):
                attrib["fill"] = self._css_color(rgbFace)
            if attrib.get("stroke", "").startswith("#"):
                attrib["stroke"] = self._css_color(gc.get_rgb())
        return attrib

    def _write_hatches(self):
        """As stock SVG, with the colors of the pattern in the figure's color space."""
        if self.color_space in (None, "srgb"):
            return super()._write_hatches()
        if not len(self._hatchd):
            return
        size = 72
        wr = self.writer
        wr.start("defs")
        for (path, face, stroke, *lw), oid in self._hatchd.values():
            # the line width is part of the key since Matplotlib 3.11
            lw = lw[0] if lw else mpl.rcParams["hatch.linewidth"]
            wr.start("pattern", id=oid, patternUnits="userSpaceOnUse", x="0", y="0",
                     width=str(size), height=str(size))
            path_data = self._convert_path(
                path, Affine2D().scale(size).scale(1.0, -1.0).translate(0, size),
                simplify=False)
            wr.element("rect", x="0", y="0", width=str(size + 1), height=str(size + 1),
                       fill="none" if face is None else self._css_color(face))
            style = {"fill": self._css_color(stroke), "stroke": self._css_color(stroke),
                     "stroke-width": str(lw), "stroke-linecap": "butt",
                     "stroke-linejoin": "miter"}
            if stroke[3] < 1:
                style["stroke-opacity"] = str(stroke[3])
            wr.element("path", d=path_data, style=_generate_css(style))
            wr.end("pattern")
        wr.end("defs")

    # -- text ---------------------------------------------------------------

    def _markup(self, s, ismath):
        """Inline XHTML/MathML for *s*.  What umbruch cannot set is an error: there is
        no quiet fallback to Matplotlib's own mathtext or usetex rendering."""
        if ismath == "TeX":
            raise NotImplementedError(
                "text.usetex is not supported by the umbruch backend; use mathtext "
                "($...$), which is converted to MathML, or rich()")
        is_rich = s.startswith(RICH_MARK)
        if is_rich:
            s = s[1:]
        esc = (lambda t: t) if is_rich else escape
        if not ismath:
            return esc(s)
        conv = config.tex_to_mathml or _default_tex_to_mathml
        out, pos = [], 0
        for m in _MATH_RE.finditer(s):
            if m.group(1).lstrip().startswith("\\mathdefault{"):
                out.append(esc(s[pos:m.start()]).replace(r"\$", "$"))
                out.append(_mathdefault_html(m.group(1).strip()))
                pos = m.end()
                continue
            try:
                ml = conv(m.group(1))
            except Exception as e:  # a converter that cannot read this TeX
                raise RuntimeError(f"cannot convert math {m.group(1)!r} to MathML: {e}") from e
            if ml is None:
                raise RuntimeError(f"cannot convert math {m.group(1)!r} to MathML: the "
                                   "converter returned None")
            out.append(esc(s[pos:m.start()]).replace(r"\$", "$"))
            out.append(_style_math(ml))
            pos = m.end()
        out.append(esc(s[pos:]).replace(r"\$", "$"))
        return "".join(out)

    def get_text_width_height_descent(self, s, prop, ismath):
        return self.bridge.measure(self._markup(s, ismath), _font_spec(prop))

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        markup = self._markup(s, ismath)
        font = _font_spec(prop)
        if font["face"]:
            self._fonts[font["face"]["alias"]] = font
        self._math = self._math or "<math" in markup
        w, h, d = self.bridge.measure(markup, font)
        asc, box_h = self.bridge.ascent(markup, font)
        alpha = gc.get_alpha() if gc.get_forced_alpha() else gc.get_rgb()[3]
        # the same declarations as the probes of the measure op, plus paint
        pad = 0.25 * font["size"]
        style = (f"margin:{_fmt(pad)}px 0 0 {_fmt(pad)}px;padding:0;"
                 f"white-space:pre;line-height:normal;"
                 f"{font['css']};color:{self._css_color(gc.get_rgb())}")
        if alpha != 1:
            style += f";opacity:{_fmt(alpha)}"
        div = f'<div xmlns="{XHTML}" style="{style}">{markup}</div>'
        _check_xml(markup)
        clip = self._get_clip_attrs(gc)
        wr = self.writer
        if clip:
            wr.start("g", **clip)
        tf = f"translate({_fmt(x)} {_fmt(y)})"
        if angle:
            tf += f" rotate({_fmt(-angle)})"
        wr.start("g", transform=tf)
        # The box top is one ascent above the baseline origin.  The island is clipped
        # to its box, so it gets a little room for overhanging glyphs.
        wr.start("foreignObject", x=_fmt(-pad), y=_fmt(-asc - pad),
                 width=_fmt(w + 2 * pad), height=_fmt(box_h + 2 * pad))
        wr.raw(div)
        wr.end("foreignObject", indent=False)
        wr.end("g")
        if clip:
            wr.end("g")

    def finalize(self):
        rules = _font_face_css(self._fonts.values(), math=self._math)
        if rules:
            self.writer.start("style")
            self.writer.data(rules)
            self.writer.end("style")
        super().finalize()

    # -- meshes -------------------------------------------------------------

    def draw_gouraud_triangles(self, gc, triangles_array, colors_array, transform):
        mode = config.gouraud
        if mode not in ("native", "auto", "subdivide", "stock"):
            raise ValueError(f"config.gouraud must be 'native', 'auto', 'subdivide' or "
                             f"'stock', not {mode!r}")
        if self.color_space not in (None, "srgb"):
            raise NotImplementedError(
                f"Gouraud shading is not supported in the {self.color_space} color space "
                "yet (umbruch paints meshes in sRGB): use shading='flat' or 'nearest', "
                "or color_space='srgb'")
        if mode == "stock":
            return super().draw_gouraud_triangles(gc, triangles_array, colors_array,
                                                  transform)
        if (mode == "native" and self.bridge.capabilities
                and "mesh_gouraud" not in self.bridge.capabilities):
            log.warning("this umbruch lacks the mesh_gouraud capability: its Gouraud "
                        "meshes are not drawn; update umbruch or set "
                        "config.gouraud = 'subdivide'")
        cols = np.asarray(colors_array, dtype=float)
        if cols.shape[-1] == 1:
            cols = np.repeat(cols, 3, axis=-1)
        if cols.shape[-1] == 3:
            cols = np.concatenate([cols, np.ones(cols.shape[:-1] + (1,))], axis=-1)
        # umbruch paints a mesh as one shading, which has one alpha
        problem = None
        if mode == "auto" and "mesh_gouraud" not in self.bridge.capabilities:
            problem = "this umbruch cannot paint meshes"
        elif np.ptp(cols[..., 3]) > 1 / 512:
            problem = "the alpha changes between the vertices"
        if mode == "native" and problem:
            raise ValueError(f"config.gouraud = 'native' is not possible: {problem}; "
                             "use config.gouraud = 'subdivide'")
        native = mode == "native" or (mode == "auto" and not problem)

        wr = self.writer
        trans = self._make_flip_transform(transform.frozen())
        wr.start("g", **self._get_clip_attrs(gc))
        if native:
            self._native_mesh(trans, np.asarray(triangles_array, dtype=float), cols)
        else:
            wr.start("g")
            for pts, c in zip(triangles_array, cols):
                self._subdivided(trans.transform(pts), c)
            wr.end("g")
        wr.end("g")

    def _native_mesh(self, trans, triangles, cols):
        """The mesh as one element: the distinct vertices (x, y, r, g, b, a) and the
        triangles as indices into them.  A mesh of a grid shares most of its vertices
        between triangles, and one element is much faster for umbruch to read than a
        path per triangle."""
        p = trans.transform(triangles.reshape(-1, 2))
        rows = np.concatenate([np.round(p, 3), np.round(cols.reshape(-1, 4), 4)], axis=1)
        rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
        vertices, index = np.unique(rows, axis=0, return_inverse=True)
        self.writer.element("g", attrib={
            "data-umbruch-mesh": "gouraud",
            "data-vertices": " ".join(map(repr, vertices.ravel().tolist())),
            "data-triangles": " ".join(map(str, index.ravel().tolist()))})

    def _tri(self, p, rgba, seam=False):
        attrib = {"fill-opacity": _fmt(rgba[3])}
        if seam and rgba[3] >= 1:
            # a hairline of the same color hides the seams that antialiasing leaves
            # between flat neighbours
            attrib.update({"stroke": self._css_color(rgba[:3]), "stroke-width": "0.4",
                           "stroke-linejoin": "round"})
        else:
            attrib["stroke"] = "none"
        self.writer.element(
            "path", d="M{} {}L{} {}L{} {}Z".format(*(_fmt(v) for v in p.ravel())),
            fill=self._css_color(rgba[:3]), attrib=attrib)

    def _subdivided(self, p, c):
        """Approximate a Gouraud triangle by flat ones: n*n sub-triangles, each in the
        color of its centroid.  n follows the color range and the size, so smooth
        large triangles get many and tiny or flat ones few."""
        span = float((c.max(axis=0) - c.min(axis=0))[:3].max())
        edge = max(np.hypot(*(p[i] - p[(i + 1) % 3])) for i in range(3))
        n = int(min(config.gouraud_max_level, max(1, np.ceil(span * 24)),
                    max(1, np.ceil(edge / 2.0))))
        if n == 1:
            return self._tri(p, c.mean(axis=0), seam=True)

        def pt(i, j):  # barycentric weights (n-i-j, i, j)/n
            w = np.array([n - i - j, i, j]) / n
            return w @ p, w @ c

        for i in range(n):
            for j in range(n - i):
                up = [pt(i, j), pt(i + 1, j), pt(i, j + 1)]
                self._tri(np.array([q[0] for q in up]),
                          np.mean([q[1] for q in up], axis=0), seam=True)
                if i + j < n - 1:
                    dn = [pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)]
                    self._tri(np.array([q[0] for q in dn]),
                              np.mean([q[1] for q in dn], axis=0), seam=True)


class FigureCanvasUmbruch(FigureCanvasSVG):
    filetypes = {"pdf": "Portable Document Format (umbruch)",
                 "svg": "SVG with umbruch extensions"}

    def get_default_filetype(self):
        return "pdf"

    def _draw_to(self, fh, bridge, bbox_inches_restore, metadata, color_space=None):
        dpi = self.figure.dpi
        self.figure.dpi = 72
        width, height = self.figure.get_size_inches()
        w, h = width * 72, height * 72
        renderer = MixedModeRenderer(
            self.figure, width, height, dpi,
            RendererUmbruch(w, h, fh, image_dpi=dpi, metadata=metadata, bridge=bridge,
                            color_space=color_space),
            bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)
        renderer.finalize()
        return w, h

    def _prefetch(self, bridge):
        """Draw to nowhere to collect the strings, measure them in one batch, repeat
        until a drawing needs nothing new (layout can change the strings, e.g. ticks)."""
        for _ in range(4):
            bridge._collect = {}
            try:
                self._draw_to(io.StringIO(), bridge, None, None)
            finally:
                pending, bridge._collect = bridge._collect, None
            if not pending:
                return
            bridge.measure_many(list(pending.values()))

    def _svg_text(self, bridge, bbox_inches_restore, metadata, color_space):
        space = color_space or config.color_space
        if space is not None:
            _space(space)
        buf = io.StringIO()
        size = self._draw_to(buf, bridge, bbox_inches_restore, metadata, space)
        return buf.getvalue(), size

    def print_svg(self, filename, *, bbox_inches_restore=None, metadata=None,
                  color_space=None, **kwargs):
        bridge = get_bridge()
        self._prefetch(bridge)
        text, _ = self._svg_text(bridge, bbox_inches_restore, metadata, color_space)
        with cbook.open_file_cm(filename, "w", encoding="utf-8") as fh:
            if not cbook.file_requires_unicode(fh):
                import codecs
                fh = codecs.getwriter("utf-8")(fh)
            fh.write(text)

    def print_pdf(self, filename, *, bbox_inches_restore=None, metadata=None,
                  color_space=None, **kwargs):
        bridge = get_bridge()
        self._prefetch(bridge)
        svg, (w, h) = self._svg_text(bridge, bbox_inches_restore, None, color_space)
        data = bridge.render(svg, w, h)
        with cbook.open_file_cm(filename, "wb") as fh:
            fh.write(data)


FigureCanvas = FigureCanvasUmbruch
