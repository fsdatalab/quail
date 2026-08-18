"""Build the one-page PDF that compares the B=1024 and B=2048 traces."""

import json
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import landscape, letter
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATHS = {
    1024: ROOT / "results/engine/torchprof_stock_B1024_streamed.json",
    2048: ROOT / "results/engine/torchprof_stock_B2048_streamed.json",
}
KERNEL_COMPARISON_PATH = (
    ROOT / "results/engine/torchprof_B1024_vs_B2048_all_kernels.json"
)
OUTPUT_PATH = (
    ROOT / "output/pdf/vllm_B1024_vs_B2048_profiler_diagnostic.pdf"
)

SWEEP = [
    (512, 60_386.5),
    (1024, 41_719.3),
    (2048, 81_983.2),
    (4096, 83_852.9),
    (8192, 92_070.6),
    (16384, 96_363.4),
    (25305, 96_877.5),
]
REPEAT_1024 = 42_081.8
INK = HexColor("#17202A")
MUTED = HexColor("#667085")
GRID = HexColor("#D9DEE7")
BLUE = HexColor("#356AE6")
LIGHT_BLUE = HexColor("#AFC6F5")
RED = HexColor("#D64545")
LIGHT_RED = HexColor("#F3B5B5")
GOLD = HexColor("#E3A52B")
GREEN = HexColor("#3A9D78")
PURPLE = HexColor("#8266CC")
TEAL = HexColor("#48A9A6")
GRAY = HexColor("#98A2B3")


def label(c, x, y, text, size=9, color=INK, font="Helvetica"):
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawString(x, y, text)


def right_label(c, x, y, text, size=9, color=INK, font="Helvetica"):
    c.setFillColor(color)
    c.setFont(font, size)
    c.drawRightString(x, y, text)


def wrapped(c, x, y, text, width, size=9, leading=12, color=MUTED):
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if stringWidth(candidate, "Helvetica", size) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for line in lines:
        label(c, x, y, line, size=size, color=color)
        y -= leading
    return y


def draw_sweep(c, x, y, w, h):
    label(c, x, y + h + 18, "Same-container sweep", 12, font="Helvetica-Bold")
    label(c, x, y + h + 5, "Prompt throughput, thousands of tokens per second", 8, MUTED)

    left, bottom = x + 42, y + 28
    plot_w, plot_h = w - 54, h - 38
    for tick in (40, 60, 80, 100):
        yy = bottom + (tick - 35) / 70 * plot_h
        c.setStrokeColor(GRID)
        c.setLineWidth(0.6)
        c.line(left, yy, left + plot_w, yy)
        right_label(c, left - 7, yy - 3, str(tick), 7, MUTED)

    points = []
    for i, (b, rate) in enumerate(SWEEP):
        xx = left + i / (len(SWEEP) - 1) * plot_w
        yy = bottom + (rate / 1000 - 35) / 70 * plot_h
        points.append((xx, yy, b, rate))

    c.setStrokeColor(BLUE)
    c.setLineWidth(1.8)
    for p1, p2 in zip(points, points[1:]):
        c.line(p1[0], p1[1], p2[0], p2[1])

    for xx, yy, b, rate in points:
        color = RED if b == 1024 else BLUE
        c.setFillColor(color)
        c.circle(xx, yy, 3.4, fill=1, stroke=0)
        label(c, xx - 11, bottom - 17, f"{b // 1024}k" if b >= 1024 else "512", 7, MUTED)
        if b == 1024:
            label(c, xx - 19, yy - 17, f"{rate / 1000:.1f}", 8, RED, "Helvetica-Bold")
            c.setFillColor(LIGHT_RED)
            c.circle(xx + 5, bottom + (REPEAT_1024 / 1000 - 35) / 70 * plot_h,
                     2.5, fill=1, stroke=0)

    label(c, left + plot_w - 23, points[-1][1] + 7,
          f"{points[-1][3] / 1000:.1f}", 8, BLUE, "Helvetica-Bold")
    label(c, left + plot_w / 2 - 48, y + 1, "max_num_batched_tokens", 8, MUTED)
    label(c, x + 6, y + 58, "The 1024 repeat was 42.1.", 8, RED)


def draw_window(c, x, y, w, h, summaries):
    label(c, x, y + h + 18, "Matched 15 second profiler windows", 12,
          font="Helvetica-Bold")
    label(c, x, y + h + 5, "Blue is time with at least one GPU kernel running", 8,
          MUTED)

    for row, batch_size in enumerate((1024, 2048)):
        summary = summaries[batch_size]
        bar_y = y + h - 34 - row * 45
        bar_x = x + 48
        bar_w = w - 48
        busy_w = bar_w * summary["gpu_busy_frac"]
        c.setFillColor(BLUE)
        c.rect(bar_x, bar_y, busy_w, 20, fill=1, stroke=0)
        c.setFillColor(LIGHT_BLUE)
        c.rect(bar_x + busy_w, bar_y, bar_w - busy_w, 20, fill=1, stroke=0)
        label(c, x, bar_y + 6, f"B={batch_size}", 8, INK, "Helvetica-Bold")
        label(c, bar_x + 5, bar_y + 6,
              f"{summary['gpu_busy_frac'] * 100:.1f}%", 8, INK,
              "Helvetica-Bold")
        right_label(c, x + w, bar_y - 13,
                    f"No kernel {100 * (1 - summary['gpu_busy_frac']):.1f}%",
                    8, MUTED)

    wrapped(c, x, y + 22,
            "B=2048 keeps the H100 busier, but both traces contain long gaps "
            "between kernels.", w, size=9, leading=12, color=INK)


def step_parts(summary):
    cycle_ms = summary["window_s"] * 1000 / summary["n_steps"]
    gpu_ms = (summary["window_s"] * summary["gpu_busy_frac"] * 1000
              / summary["n_steps"])
    inside_gap_ms = max(0, summary["step_mean_ms"] - gpu_ms)
    between_ms = max(0, cycle_ms - summary["step_mean_ms"])
    return cycle_ms, gpu_ms, inside_gap_ms, between_ms


def draw_step(c, x, y, w, summaries):
    label(c, x, y + 80, "Average profiled step on one time scale", 12,
          font="Helvetica-Bold")
    label(c, x, y + 67,
          "B=2048 processes twice the tokens, while its step is only 31% longer",
          8, MUTED)
    bar_x, bar_w, max_ms = x + 62, w - 150, 70
    for row, batch_size in enumerate((1024, 2048)):
        summary = summaries[batch_size]
        cycle_ms, gpu_ms, inside_gap_ms, between_ms = step_parts(summary)
        bar_y = y + 36 - row * 31
        parts = [
            (gpu_ms, BLUE, f"{gpu_ms:.1f} ms kernel"),
            (inside_gap_ms, LIGHT_BLUE, f"{inside_gap_ms:.1f} ms no kernel"),
            (between_ms, GRAY, ""),
        ]
        label(c, x, bar_y + 5, f"B={batch_size}", 8, INK, "Helvetica-Bold")
        cursor = bar_x
        for value, color, value_text in parts:
            part_w = bar_w * value / max_ms
            c.setFillColor(color)
            c.rect(cursor, bar_y, part_w, 20, fill=1, stroke=0)
            if value_text and part_w > 65:
                label(c, cursor + 4, bar_y + 6, value_text, 7, INK,
                      "Helvetica-Bold")
            cursor += part_w
        right_label(c, x + w, bar_y + 5,
                    f"{summary['step_mean_ms']:.1f} ms step, "
                    f"{cycle_ms:.1f} ms cycle", 8, MUTED)


def draw_kernel_mix(c, x, y, w, kernel_comparison):
    label(c, x, y + 58, "GPU work mix is almost unchanged", 12,
          font="Helvetica-Bold")
    label(c, x, y + 45,
          "The regression is not explained by a different kind of GPU kernel",
          8, MUTED)
    family_rows = {row["family"]: row
                   for row in kernel_comparison["kernel_families"]}
    category_info = [
        (("DeepGEMM FP8 matrix multiplication",
          "Output projection matrix multiplication"),
         "Matrix multiplication", BLUE),
        (("FP8 activation quantization", "Static FP8 quantization"),
         "FP8 quantization", GOLD),
        (("FlashAttention forward", "FlashAttention combine",
          "FlashAttention metadata", "KV cache write"),
         "Attention and KV", TEAL),
        (("Fused add and RMS normalization",
          "Fused add, index, and RMS normalization",
          "Embedding RMS normalization"),
         "Normalization", GREEN),
        (("SiLU and gated multiply",), "SiLU and multiply", PURPLE),
        (("Triton fused reduction 3",), "Other fused reduction", RED),
    ]
    bar_x, bar_w = x + 62, w - 62
    for row, batch_size in enumerate((1024, 2048)):
        bar_y = y + 20 - row * 25
        label(c, x, bar_y + 5, f"B={batch_size}", 8, INK, "Helvetica-Bold")
        cursor = bar_x
        used_frac = 0
        for families, _name, color in category_info:
            frac = sum(
                family_rows[family][f"B{batch_size}"]["pct_gpu_kernel_time"]
                / 100 for family in families
            )
            used_frac += frac
            part_w = bar_w * frac
            c.setFillColor(color)
            c.rect(cursor, bar_y, part_w, 18, fill=1, stroke=0)
            if part_w > 38:
                label(c, cursor + 3, bar_y + 5, f"{frac * 100:.1f}%", 7, INK,
                      "Helvetica-Bold")
            cursor += part_w
        other_frac = max(0, 1 - used_frac)
        c.setFillColor(GRAY)
        c.rect(cursor, bar_y, bar_w * other_frac, 18, fill=1, stroke=0)

    legend_x = x
    for _families, name, color in category_info + [
            ((), "Remaining kernels", GRAY)]:
        c.setFillColor(color)
        c.rect(legend_x, y - 20, 7, 7, fill=1, stroke=0)
        label(c, legend_x + 10, y - 19, name, 7, MUTED)
        legend_x += stringWidth(name, "Helvetica", 7) + 25


def build():
    summaries = {}
    for batch_size, path in SUMMARY_PATHS.items():
        with open(path) as f:
            summaries[batch_size] = json.load(f)
    with open(KERNEL_COMPARISON_PATH) as f:
        kernel_comparison = json.load(f)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    page_w, page_h = landscape(letter)
    c = canvas.Canvas(str(OUTPUT_PATH), pagesize=(page_w, page_h))
    c.setTitle("vLLM B=1024 and B=2048 profiler diagnostic")
    margin = 42

    label(c, margin, page_h - 42, "Why B = 1024 is slow in vLLM 0.26.0", 20,
          INK, "Helvetica-Bold")
    label(c, margin, page_h - 59,
          "Qwen3-4B-FP8, 10,000 documents, one H100, one container, engine cleared between runs",
          9, MUTED)

    gap = 32
    top_w = (page_w - 2 * margin - gap) / 2
    draw_sweep(c, margin, 337, top_w, 150)
    draw_window(c, margin + top_w + gap, 337, top_w, 150, summaries)
    draw_step(c, margin, 218, page_w - 2 * margin, summaries)
    draw_kernel_mix(c, margin, 117, page_w - 2 * margin, kernel_comparison)

    c.setStrokeColor(GRID)
    c.line(margin, 69, page_w - margin, 69)
    label(c, margin, 48,
          "Next test: at B=1024, skip the FP8 size check and call DeepGEMM directly. "
          "That test will show whether the wrapper causes the gaps.",
          9, INK, "Helvetica-Bold")
    label(c, margin, 31,
          "Sources: same-container sweep on 2026-08-12; matched rank 0 torch-profiler "
          "traces with 307 steps at B=1024 and 226 steps at B=2048.",
          7, MUTED)
    c.save()
    print(OUTPUT_PATH)


if __name__ == "__main__":
    build()
