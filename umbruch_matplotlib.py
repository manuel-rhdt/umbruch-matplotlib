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
* Gouraud triangles are drawn as flat sub-triangles (``config.gouraud ==
  "subdivide"``) because umbruch does not read mesh shadings from SVG yet.
  ``"native"`` writes exact vertex colors for a renderer that does.
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
import json
import os
import re
import shutil
import subprocess
import atexit
import logging
import struct
import tempfile
import threading
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import numpy as np

import matplotlib as mpl
from matplotlib import cbook, font_manager as fm
from matplotlib.backend_bases import FigureManagerBase
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_svg import (
    FigureCanvasSVG, RendererSVG, _XMLWriter)
from matplotlib.colors import rgb2hex

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
    #: found by ``find_umbruch``.
    bridge = None
    #: callable(tex: str) -> MathML string, or None to use latex2mathml (an error
    #: if that is not installed).
    tex_to_mathml = None
    #: "auto": "native" if the renderer reports the mesh_gouraud capability, else
    #: "subdivide"; "subdivide": flat sub-triangles (works with any renderer);
    #: "native": exact vertex colors in data-colors, for a renderer that writes a
    #: type 4 shading; "stock": Matplotlib's three-gradient filter emulation.
    gouraud = "auto"
    gouraud_max_level = 8
    #: Font of ``$...$`` math: a family name (umbruch's own "STIX Two Math" and "Latin
    #: Modern Math", or any OpenType math font Matplotlib finds) or the path of a font
    #: file.  None follows ``mathtext.fontset``: "cm" is Latin Modern Math, anything
    #: else STIX Two Math.
    math_font = None


config = _Config()


class UmbruchBridge:
    """What the backend needs from a renderer.  Units are SVG user units, which are
    CSS px inside a foreignObject and, in Matplotlib's SVG, points."""

    #: If False, text is delegated to the stock SVG renderer.
    handles_text = True

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
    """Bridge to one long-lived ``umbruch --serve`` process (protocol version 1).

    Measuring and rendering are requests over stdio, framed as in LSP.  ``measure``
    returns baseline, line box, advance width and ink bounds, so Matplotlib is told
    the ink extents directly; the line box is only used to place the text.  A server
    that exited (panic, timeout) is started again and the idempotent request is
    repeated once.
    """

    PROTOCOL = 1
    BATCH = 2000

    def __init__(self, executable="umbruch", args=("--no-system-fonts",)):
        super().__init__()
        self.executable = executable
        self.args = list(args)
        self._proc = None
        self._lock = threading.RLock()
        self._next_id = 0
        self._pid = None
        atexit.register(self.close)

    # -- process and framing -------------------------------------------------

    def _start(self):
        self._proc = subprocess.Popen(
            [self.executable, *self.args, "--serve"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
        try:
            hello, _ = self._exchange("hello", {"protocol": self.PROTOCOL})
        except Exception:
            self._kill()
            raise
        if hello.get("protocol") != self.PROTOCOL:
            self._kill()
            raise RuntimeError(f"umbruch speaks protocol {hello.get('protocol')}, "
                               f"expected {self.PROTOCOL}")
        self.capabilities = frozenset(hello.get("capabilities", ()))
        missing = {"measure", "render"} - self.capabilities
        if missing:
            self._kill()
            raise RuntimeError(f"umbruch --serve lacks {sorted(missing)}")
        self.version = hello.get("umbruch")
        self._pid = os.getpid()

    def _kill(self):
        p, self._proc = self._proc, None
        if p is not None:
            try:
                p.kill()
            except OSError:
                pass
            p.wait()

    def _read_exact(self, n):
        data = self._proc.stdout.read(n)
        if len(data) != n:
            raise EOFError("umbruch server closed the stream")
        return data

    def _exchange(self, op, params):
        self._next_id += 1
        body = json.dumps({"id": self._next_id, "op": op, "params": params},
                          ensure_ascii=False).encode("utf-8")
        self._proc.stdin.write(b"Content-Length: %d\r\n\r\n" % len(body) + body)
        self._proc.stdin.flush()
        length = None
        while True:
            line = self._proc.stdout.readline()
            if not line:
                raise EOFError("umbruch server closed the stream")
            line = line.strip()
            if not line:
                break
            name, _, value = line.decode("ascii", "replace").partition(":")
            if name.lower() == "content-length":
                length = int(value)
        if length is None:
            raise RuntimeError("umbruch server sent a frame without Content-Length")
        reply = json.loads(self._read_exact(length))
        if reply.get("id") != self._next_id:
            raise RuntimeError(f"umbruch server answered request {reply.get('id')}, "
                               f"expected {self._next_id}")
        if not reply.get("ok"):
            e = reply.get("error", {})
            raise RuntimeError(f"umbruch {op} failed: {e.get('code')}: {e.get('message')}")
        result = reply["result"]
        attachment = self._read_exact(result["attachment_length"]) \
            if result.get("attachment_length") else b""
        return result, attachment

    def request(self, op, params):
        with self._lock:
            if self._proc is not None and self._pid != os.getpid():
                # forked child: the pipes belong to the parent's conversation
                for f in (self._proc.stdin, self._proc.stdout):
                    try:
                        f.close()
                    except OSError:
                        pass
                self._proc = None
            for attempt in (0, 1):
                try:
                    if self._proc is None or self._proc.poll() is not None:
                        self._start()
                    return self._exchange(op, params)
                except (EOFError, BrokenPipeError, OSError):
                    self._kill()
                    if attempt:
                        raise RuntimeError("the umbruch server keeps exiting") from None

    # -- bridge --------------------------------------------------------------

    def measure_many(self, items):
        items = [it for it in dict((self.key(m, f), (m, f)) for m, f in items).values()
                 if self.key(*it) not in self._cache]
        for start in range(0, len(items), self.BATCH):
            chunk = items[start:start + self.BATCH]
            # @font-face rules apply to the probes of the same request
            rules = _font_face_css((f for _, f in chunk),
                                   math=any("<math" in m for m, _ in chunk))
            head = [{"markup": f"<style>{rules}</style>", "css": ""}] if rules else []
            result, _ = self.request("measure", {"probes": head + [
                {"markup": m, "css": f["css"]} for m, f in chunk]})
            for (markup, font), m in zip(chunk, result["metrics"][len(head):]):
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
        result, pdf = self.request("render", {
            "document": {"source": svg, "kind": "xml"}, "options": {}})
        for w in result.get("warnings", ()):
            if "no `lang`" not in str(w):  # PDF/UA is not a goal of a plot
                log.warning("umbruch: %s", w)
        return pdf

    def close(self):
        with self._lock:
            p, self._proc = self._proc, None
        if p is None:
            return
        try:
            self._proc = p
            self._exchange("shutdown", {})
        except Exception:
            pass
        finally:
            self._proc = None
            try:
                p.stdin.close()
                p.wait(timeout=5)
            except Exception:
                p.kill()


def find_umbruch():
    """The umbruch executable: $UMBRUCH, the one installed with the ``umbruch`` package,
    or ``umbruch`` on PATH."""
    exe = os.environ.get("UMBRUCH")
    if exe:
        return exe
    try:
        import umbruch
        exe = umbruch.find_umbruch()
    except ImportError:
        exe = None
    return exe or shutil.which("umbruch")


def get_bridge():
    """The bridge to the umbruch server (started on first use).  There is no fallback to
    another way of running umbruch: if it cannot be found or started, this raises."""
    if config.bridge is None:
        exe = find_umbruch()
        if exe is None:
            raise RuntimeError("umbruch executable not found: pip install umbruch, put it on "
                               "PATH, or set $UMBRUCH (or umbruch_matplotlib.config.bridge)")
        server = UmbruchServer(exe)
        server._start()
        config.bridge = server
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
    def raw(self, s):
        # Pretty-printing would insert whitespace between inline elements.
        self._XMLWriter__flush()
        self._XMLWriter__write(s)


class RendererUmbruch(RendererSVG):
    def __init__(self, width, height, svgwriter, basename=None, image_dpi=72,
                 *, metadata=None, bridge=None):
        super().__init__(width, height, svgwriter, basename, image_dpi,
                         metadata=metadata)
        self.writer.__class__ = _RawWriter
        self.bridge = bridge
        self._fonts = {}
        self._math = False

    # -- text ---------------------------------------------------------------

    def _handles_text(self):
        return self.bridge is not None and self.bridge.handles_text

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

    def _degrade(self, s):
        """Without umbruch text support, rich markup is reduced to plain text."""
        if isinstance(s, str) and s.startswith(RICH_MARK):
            return _TAG_RE.sub("", s[1:])
        return s

    def get_text_width_height_descent(self, s, prop, ismath):
        if not self._handles_text():
            return super().get_text_width_height_descent(self._degrade(s), prop, ismath)
        return self.bridge.measure(self._markup(s, ismath), _font_spec(prop))

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if not self._handles_text():
            return super().draw_text(gc, x, y, self._degrade(s), prop, angle,
                                     ismath, mtext)
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
                 f"{font['css']};color:{rgb2hex(gc.get_rgb())}")
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
        if mode == "auto":
            mode = "native" if (self.bridge is not None and
                                "mesh_gouraud" in self.bridge.capabilities) else "subdivide"
        if self.bridge is None or mode == "stock":
            return super().draw_gouraud_triangles(gc, triangles_array, colors_array,
                                                  transform)
        wr = self.writer
        trans = self._make_flip_transform(transform.frozen())
        cols = np.asarray(colors_array, dtype=float)
        if cols.shape[-1] == 1:
            cols = np.repeat(cols, 3, axis=-1)
        if cols.shape[-1] == 3:
            cols = np.concatenate([cols, np.ones(cols.shape[:-1] + (1,))], axis=-1)
        wr.start("g", **self._get_clip_attrs(gc))
        wr.start("g", attrib={"data-umbruch-mesh": "gouraud"})
        for pts, c in zip(triangles_array, cols):
            p = trans.transform(pts)
            if mode == "native":
                # exact vertex colors for a renderer that writes a type 4 shading
                self._tri(p, c.mean(axis=0), extra={"data-colors": ";".join(
                    ",".join(_fmt(v) for v in col) for col in c)})
            else:
                self._subdivided(p, c)
        wr.end("g")
        wr.end("g")

    def _tri(self, p, rgba, extra=None, seam=False):
        attrib = {"fill-opacity": _fmt(rgba[3]), **(extra or {})}
        if seam and rgba[3] >= 1:
            # a hairline of the same color hides the seams that antialiasing leaves
            # between flat neighbours
            attrib.update({"stroke": rgb2hex(rgba[:3]), "stroke-width": "0.4",
                           "stroke-linejoin": "round"})
        else:
            attrib["stroke"] = "none"
        self.writer.element(
            "path", d="M{} {}L{} {}L{} {}Z".format(*(_fmt(v) for v in p.ravel())),
            fill=rgb2hex(rgba[:3]), attrib=attrib)

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

    def _draw_to(self, fh, bridge, bbox_inches_restore, metadata):
        dpi = self.figure.dpi
        self.figure.dpi = 72
        width, height = self.figure.get_size_inches()
        w, h = width * 72, height * 72
        renderer = MixedModeRenderer(
            self.figure, width, height, dpi,
            RendererUmbruch(w, h, fh, image_dpi=dpi, metadata=metadata, bridge=bridge),
            bbox_inches_restore=bbox_inches_restore)
        self.figure.draw(renderer)
        renderer.finalize()
        return w, h

    def _prefetch(self, bridge):
        """Draw to nowhere to collect the strings, measure them in one batch, repeat
        until a drawing needs nothing new (layout can change the strings, e.g. ticks)."""
        if not bridge.handles_text:
            return
        for _ in range(4):
            bridge._collect = {}
            try:
                self._draw_to(io.StringIO(), bridge, None, None)
            finally:
                pending, bridge._collect = bridge._collect, None
            if not pending:
                return
            bridge.measure_many(list(pending.values()))

    def print_svg(self, filename, *, bbox_inches_restore=None, metadata=None,
                  **kwargs):
        bridge = get_bridge()
        self._prefetch(bridge)
        with cbook.open_file_cm(filename, "w", encoding="utf-8") as fh:
            if not cbook.file_requires_unicode(fh):
                import codecs
                fh = codecs.getwriter("utf-8")(fh)
            self._draw_to(fh, bridge, bbox_inches_restore, metadata)

    def print_pdf(self, filename, *, bbox_inches_restore=None, metadata=None,
                  **kwargs):
        bridge = get_bridge()
        self._prefetch(bridge)
        buf = io.StringIO()
        w, h = self._draw_to(buf, bridge, bbox_inches_restore, None)
        data = bridge.render(buf.getvalue(), w, h)
        with cbook.open_file_cm(filename, "wb") as fh:
            fh.write(data)


FigureCanvas = FigureCanvasUmbruch
