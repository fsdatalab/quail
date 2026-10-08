r"""Plot the classification profile windows and capture their Perfetto views.

Reads the torch.profiler windows that experiments/cells/classify_profiles.py
saved for IMDB-11 and IMDB-14 on three models, Quail and stock vLLM.
Writes two PNG plots, the GPU busy share of the profiled windows and
the longest GPU idle gap in them, and, with --screenshots, one Perfetto
screenshot per window in SHOTS. Pull the two run directories,
then run from the repository root:

    W=/tmp/classify-profiles; mkdir -p "$W"
    P=ablations/classify-profiles
    uv run modal volume get quail-results "$P/20261008T042520Z" "$W/"
    uv run modal volume get quail-results "$P/20261008T052000Z" "$W/"
    uv run --with matplotlib --with playwright==1.55.0 \
      python reports/make_classify_profiles_plots.py "$W" --screenshots

A run in both directories is read from the later one. Screenshots load
each trace into https://ui.perfetto.dev in Chromium; set CHROMIUM to the
browser executable if Playwright's own is not installed.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from plot_colors import BLUE, ORANGE

HERE = Path(__file__).resolve().parent
OUT = HERE / "plots"
TAGS = ["20261008T042520Z", "20261008T052000Z"]
VOLUME_ROOT = "/results/ablations/classify-profiles/"
MODELS = [("qwen3-4b-fp8", "Qwen3 4B FP8"),
          ("diffusion-gemma-26b-a4b-fp8", "DiffusionGemma 26B-A4B FP8"),
          ("decision-2.0-kai-0.6b-bf16", "Decision-2.0-Kai 0.6B BF16")]
QUERIES = ["IMDB-11", "IMDB-14"]
METHODS = [("quail", "Quail", BLUE),
           ("stock_vllm", "Stock vLLM (operator-at-a-time)", ORANGE)]
# stock vLLM's planner refuses AI.CLASSIFY on this model
REFUSED = {("diffusion-gemma-26b-a4b-fp8", "stock_vllm")}
# (model, backend, query, window index): pairs of the same model and
# query, plus DiffusionGemma, which has no stock vLLM run
SHOTS = [
    ("decision-2.0-kai-0.6b-bf16", "stock_vllm", "IMDB-11", 0),
    ("decision-2.0-kai-0.6b-bf16", "quail", "IMDB-11", 0),
    ("decision-2.0-kai-0.6b-bf16", "stock_vllm", "IMDB-14", 0),
    ("decision-2.0-kai-0.6b-bf16", "quail", "IMDB-14", 0),
    ("qwen3-4b-fp8", "stock_vllm", "IMDB-11", 0),
    ("qwen3-4b-fp8", "quail", "IMDB-11", 0),
    ("qwen3-4b-fp8", "stock_vllm", "IMDB-14", 0),
    ("qwen3-4b-fp8", "quail", "IMDB-14", 0),
    ("diffusion-gemma-26b-a4b-fp8", "quail", "IMDB-11", 0),
    ("diffusion-gemma-26b-a4b-fp8", "quail", "IMDB-14", 0),
]
SHORT = {"qwen3-4b-fp8": "qwen3_4b", "diffusion-gemma-26b-a4b-fp8": "dgemma",
         "decision-2.0-kai-0.6b-bf16": "kai"}
WIDTH = 0.38


def windows(record):
    """Return a run's windows in the cell's current layout.

    The first runs saved one window under "window" with its GPU summary
    at the top level; later runs save a "windows" list.
    """
    if "windows" in record:
        return record["windows"]
    window = record["window"]
    if not window.get("armed"):
        return []
    return [{"index": 0, "start_s": window["start_s"], "end_s": window["end_s"],
             "passes": window["passes"], "cut": window["cut"],
             "gpu": record.get("gpu"), "kernels": record.get("kernels"),
             "trace": record.get("trace")}]


def load(workdir):
    """Read each run's window.json, the later directory first."""
    runs = {}
    for tag in TAGS:
        for path in sorted((workdir / tag).glob("*/*/*/window.json")):
            model, backend, query = path.parent.relative_to(workdir / tag).parts
            if query not in QUERIES:
                continue
            record = json.loads(path.read_text())
            record["windows"] = windows(record)
            if record["windows"]:
                runs[(model, backend, query)] = record
    return runs


def measures(record):
    """GPU busy share and longest idle gap over a run's full windows.

    The busy share is GPU-active seconds over the seconds from each
    window's first device event to its last, summed over the windows. A
    window the query's end cut short is left out.
    """
    full = [w for w in record["windows"] if not w["cut"] and w.get("gpu")]
    busy = sum(w["gpu"]["busy_s"] for w in full)
    span = sum(w["gpu"]["span_s"] for w in full)
    gap = max((w["gpu"]["gaps"][0]["ms"] for w in full if w["gpu"]["gaps"]),
              default=0.0)
    return {"busy": 100 * busy / span, "gap_ms": gap, "windows": len(full)}


def positions():
    """Each (model, query) group's x position, with a gap between models."""
    xs, centers, x = [], [], 0.0
    for model, name in MODELS:
        start = x
        for query in QUERIES:
            xs.append((model, query, x))
            x += 1.0
        centers.append((name, (start + x - 1.0) / 2))
        x += 0.7
    return xs, centers


def draw(axis, values, key, unit, log=False):
    """Draw grouped bars per model and query; an x marks a refused run."""
    xs, centers = positions()
    for offset, (backend, _, color) in zip((-WIDTH / 2, WIDTH / 2), METHODS):
        for model, query, x in xs:
            value = values.get((model, backend, query))
            if value is None:
                axis.plot(x + offset, 0.03, marker="x", color=color,
                          markersize=7, linestyle="none", clip_on=False,
                          transform=axis.get_xaxis_transform())
                continue
            v = value[key]
            axis.bar(x + offset, v, width=WIDTH * 0.92, color=color,
                     edgecolor="none")
            text = (f"{v:.0f}" if key == "busy" or v >= 1
                    else f"{v:.2f}")
            axis.annotate(text, (x + offset, v), xytext=(0, 3),
                          textcoords="offset points", ha="center",
                          va="bottom", fontsize=8.5, color="#333333")
    axis.set_xticks([x for _, _, x in xs], [query for _, query, _ in xs])
    if log:
        axis.set_yscale("log")
        axis.set_ylim(0.005, 3000)
        axis.set_ylabel(f"{unit} (log scale)")
    else:
        axis.set_ylim(0, 112)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.set_ylabel(unit)
    for name, center in centers:
        axis.annotate(name, (center, 0), xycoords=("data", "axes fraction"),
                      xytext=(0, -30), textcoords="offset points",
                      ha="center", va="top", fontsize=9.5, color="#555555",
                      annotation_clip=False)
    axis.grid(axis="y", color="#eeeeee", linewidth=0.6)
    axis.set_axisbelow(True)


def legend(axis):
    """Place the engine legend under the title."""
    handles = [Patch(facecolor=color, label=label)
               for _, label, color in METHODS]
    handles.append(Line2D([0], [0], marker="x", color=ORANGE, linestyle="none",
                          label="Stock vLLM refuses AI.CLASSIFY on this model"))
    axis.legend(handles=handles, loc="lower left", ncol=3,
                bbox_to_anchor=(0, 1.0), fontsize=9)


def plot(values):
    """Write the busy-share plot and the busy-share-and-gap plot."""
    title = ("GPU busy share of profiled 5 s windows, QUAIL-B IMDB-11 "
             "and IMDB-14, sf 0.5, one H100")
    figure, axis = plt.subplots(figsize=(11, 4.6))
    draw(axis, values, "busy", "percent")
    axis.set_title(title, pad=36)
    legend(axis)
    figure.subplots_adjust(left=0.07, right=0.99, top=0.8, bottom=0.2)
    figure.savefig(OUT / "classify_profiles_gpu_busy.png", dpi=300)
    plt.close(figure)

    figure, axes = plt.subplots(2, 1, figsize=(11, 8.2))
    draw(axes[0], values, "busy", "percent")
    axes[0].set_title(title, pad=36)
    legend(axes[0])
    draw(axes[1], values, "gap_ms", "milliseconds", log=True)
    axes[1].set_title("Longest GPU idle gap in the same windows")
    figure.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.1,
                           hspace=0.75)
    figure.savefig(OUT / "classify_profiles_gpu_busy_and_gap.png", dpi=300)
    plt.close(figure)


def shot_title(model, backend, query, window):
    """The one-line title stamped above a screenshot."""
    engine = dict((key, label) for key, label, _ in METHODS)[backend]
    name = dict(MODELS)[model]
    gpu = window["gpu"]
    busy = 100 * gpu["busy_s"] / gpu["span_s"]
    return (f"{engine}, {name}, {query}, 5 s window at "
            f"{window['start_s']:.0f} s: GPU busy {busy:.0f}%")


def screenshots(workdir, runs):
    """Load each SHOTS trace into Perfetto and save its timeline."""
    import matplotlib
    from PIL import Image, ImageDraw, ImageFont
    from playwright.sync_api import sync_playwright

    fonts = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
    font = ImageFont.truetype(str(fonts / "DejaVuSans-Bold.ttf"), 52)
    executable = os.environ.get("CHROMIUM")
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=executable, args=["--no-sandbox"])
        for model, backend, query, index in SHOTS:
            window = next(w for w in runs[(model, backend, query)]["windows"]
                          if w["index"] == index)
            trace = workdir / window["trace"].removeprefix(VOLUME_ROOT)
            page = browser.new_page(viewport={"width": 1800, "height": 1150},
                                    device_scale_factor=2)
            page.goto("https://ui.perfetto.dev/", timeout=120_000)
            page.wait_for_timeout(8000)
            page.evaluate(
                """data => {
                    const bytes = Uint8Array.from(atob(data), c => c.charCodeAt(0));
                    window.postMessage({perfetto: {buffer: bytes.buffer,
                                                   title: 'trace'}}, '*');
                }""",
                __import__("base64").b64encode(trace.read_bytes()).decode())
            page.wait_for_selector(".pf-track__shell", timeout=240_000)
            page.wait_for_timeout(12_000)
            page.locator('button[title="Hide sidebar"]').first.click()
            page.locator('button[title="Expand all"]').first.click()
            page.wait_for_timeout(6000)
            # the timeline down to the last GPU stream track
            bottom = page.evaluate(
                """() => Math.max(...[...document.querySelectorAll(
                    '.pf-track__shell')].filter(s => /^stream/.test(
                    s.querySelector('.pf-track__title')?.textContent ?? ''))
                    .map(s => s.getBoundingClientRect().bottom))""")
            path = OUT / (f"classify_profiles_{SHORT[model]}_{query.lower()}_"
                          f"{backend}.png")
            page.screenshot(path=str(path),
                            clip={"x": 0, "y": 112, "width": 1800,
                                  "height": bottom + 6 - 112})
            page.close()
            image = Image.open(path).convert("RGB")
            band = 110
            framed = Image.new("RGB", (image.width, image.height + band),
                               "white")
            ImageDraw.Draw(framed).text(
                (36, 28), shot_title(model, backend, query, window),
                font=font, fill="#222222")
            framed.paste(image, (0, band))
            framed.save(path, dpi=(300, 300))
            print(f"{path.name}: {shot_title(model, backend, query, window)}")
        browser.close()


def main():
    """Write the plots and, with --screenshots, the Perfetto views."""
    workdir = Path(sys.argv[1])
    plt.style.use(HERE / "quail.mplstyle")
    plt.rcParams["figure.autolayout"] = False
    OUT.mkdir(exist_ok=True)
    runs = load(workdir)
    values = {key: measures(record) for key, record in runs.items()
              if key[:2] not in REFUSED}
    for (model, backend, query), value in sorted(values.items()):
        print(f"{model:28s} {backend:10s} {query:8s} busy {value['busy']:5.1f}% "
              f"longest gap {value['gap_ms']:7.1f} ms "
              f"over {value['windows']} windows")
    plot(values)
    if "--screenshots" in sys.argv[2:]:
        screenshots(workdir, runs)


if __name__ == "__main__":
    main()
