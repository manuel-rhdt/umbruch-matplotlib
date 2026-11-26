//! `hayro-raster [--dpi N] [--page N] IN.pdf OUT.png`
//!
//! Renders one page of a PDF to a PNG with hayro. `-` stands for standard input or output.
//! The background is transparent (what is opaque is drawn by the document itself). The image
//! is `round(width_pt * dpi / 72)` by `round(height_pt * dpi / 72)` pixels, with the page
//! scaled to fill it exactly, so a 6.4 x 4.8 in figure at 100 dpi is 640 x 480 pixels.

use std::io::{Read, Write};
use std::process::ExitCode;

use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_interpret::util::TransformExt;
use hayro::hayro_syntax::Pdf;
use hayro::kurbo::Affine;
use hayro::vello_cpu::color::palette::css::TRANSPARENT;
use hayro::vello_cpu::{Pixmap, RasterizerSettings, RenderContext, Resources, TargetInit};
use hayro::{RenderCache, RenderSettings, render_into};

fn usage() -> ! {
    eprintln!("usage: hayro-raster [--dpi N] [--page N] IN.pdf OUT.png   (`-`: stdin / stdout)");
    std::process::exit(2)
}

fn fail(msg: impl std::fmt::Display) -> ! {
    eprintln!("hayro-raster: {msg}");
    std::process::exit(1)
}

fn main() -> ExitCode {
    let mut dpi = 72.0f64;
    let mut page_no = 1usize;
    let mut files = Vec::new();
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dpi" => dpi = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "--page" => page_no = args.next().and_then(|v| v.parse().ok()).unwrap_or_else(|| usage()),
            "-h" | "--help" => usage(),
            _ => files.push(a),
        }
    }
    let [input, output] = <[String; 2]>::try_from(files).unwrap_or_else(|_| usage());
    if !(dpi.is_finite() && dpi > 0.0) || page_no == 0 {
        usage()
    }

    let data = if input == "-" {
        let mut v = Vec::new();
        std::io::stdin().read_to_end(&mut v).unwrap_or_else(|e| fail(e));
        v
    } else {
        std::fs::read(&input).unwrap_or_else(|e| fail(format!("{input}: {e}")))
    };
    let pdf = Pdf::new(data).unwrap_or_else(|e| fail(format!("not a readable PDF: {e:?}")));
    let pages = pdf.pages();
    let page = pages
        .get(page_no - 1)
        .unwrap_or_else(|| fail(format!("page {page_no} of {}", pages.len())));

    let (w_pt, h_pt) = page.render_dimensions();
    let scale = dpi / 72.0;
    let (w, h) = ((w_pt as f64 * scale).round(), (h_pt as f64 * scale).round());
    if !(1.0..=65535.0).contains(&w) || !(1.0..=65535.0).contains(&h) {
        fail(format!("image of {w} x {h} pixels is out of range (1 to 65535)"));
    }
    let (sx, sy) = (w / w_pt as f64, h / h_pt as f64);

    let mut ctx = RenderContext::new(w as u16, h as u16);
    let transform = Affine::scale_non_uniform(sx, sy) * page.initial_transform(true).to_kurbo();
    render_into(
        page,
        &RenderCache::new(),
        &InterpreterSettings::default(),
        &RenderSettings::default(),
        &mut ctx,
        transform,
    );
    ctx.flush();
    let mut pixmap = Pixmap::new(ctx.width(), ctx.height());
    ctx.render_with(
        &mut pixmap,
        &mut Resources::default(),
        RasterizerSettings { target_init: TargetInit::Clear(TRANSPARENT), ..Default::default() },
    );
    let png = pixmap.into_png().unwrap_or_else(|e| fail(format!("PNG encoding: {e:?}")));

    if output == "-" {
        std::io::stdout().write_all(&png).unwrap_or_else(|e| fail(e));
    } else {
        std::fs::write(&output, png).unwrap_or_else(|e| fail(format!("{output}: {e}")));
    }
    ExitCode::SUCCESS
}
