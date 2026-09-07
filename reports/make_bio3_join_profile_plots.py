"""Plot the BIO-3 CPU flame graph from saved quail-results summaries.

    W=/tmp/quail-bio3-vllm-profile
    mkdir -p "$W"
    RUN=ablations/vllm-join-profile-20260907T013304Z
    uv run modal volume get quail-results "$RUN/result.json" "$W/result.json"
    uv run modal volume get quail-results "$RUN/cpu-flamegraph.json" "$W/cpu-flamegraph.json"
    uv run --with matplotlib python reports/make_bio3_join_profile_plots.py "$W"

The summary aggregates nested CPU intervals from join-0/worker.trace.json.gz
within the join annotation in join-0/driver.trace.json.gz, using
experiments.profile_flamegraph.read_cpu_flamegraph. Names are unchanged.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from plot_colors import BLUE, DARK, GRAY, GREEN, ORANGE


HERE = Path(__file__).resolve().parent


def layout(node, start=0, depth=0):
    """Yield aggregate intervals with children placed inside their parent."""
    yield start, depth, node
    for child in node["children"]:
        yield from layout(child, start, depth + 1)
        start += child["seconds"]


def color(node):
    """Return the color for a recorded operation category."""
    if node["name"].startswith("vllm.scheduler."):
        return ORANGE
    if node["name"].startswith("[") or node["name"] == "Worker main thread":
        return GRAY
    return BLUE


def plot_flamegraph(root):
    """Save a static view of the largest recorded CPU operations."""
    plt.style.use(HERE / "quail.mplstyle")
    visible = [(start, depth, node) for start, depth, node in layout(root)
               if depth <= 7 and node["seconds"] >= root["seconds"] / 1500]
    levels = max(depth for _, depth, _ in visible) + 1
    fig, ax = plt.subplots(figsize=(17, max(6, 2.4 + 0.45 * levels)))
    fig.set_layout_engine("none")
    total = root["seconds"]
    for start, depth, node in visible:
        patch = Rectangle((start, depth), node["seconds"], 0.92,
                          facecolor=color(node), edgecolor="white", linewidth=0.4)
        ax.add_patch(patch)
        ax.add_patch(Rectangle((start, depth + 0.78), node["seconds"], 0.14,
                               facecolor=DARK, linewidth=0))
        ax.add_patch(Rectangle((start, depth + 0.78), node["gpu_seconds"], 0.14,
                               facecolor=GREEN, linewidth=0))
        width_fraction = node["seconds"] / total
        if width_fraction > 0.055:
            limit = max(4, int(width_fraction * 170))
            name = node["name"]
            label = name if len(name) <= limit else name[:limit - 3] + "..."
            text = ax.text(start + node["seconds"] / 2, depth + 0.37,
                           f"{label}\n{node['seconds']:.2f} s elapsed\nGPU {node['gpu_seconds']:.2f} s",
                           ha="center", va="center",
                           fontsize=9, color="black", clip_on=True)
            text.set_clip_path(patch)
    ax.set(xlim=(0, total), ylim=(levels, -0.2), xlabel="seconds", yticks=[])
    ax.set_title("BIO-3, pipelined vLLM, CPU intervals and concurrent GPU activity", pad=18)
    fig.text(0.06, 0.05,
             "Width is total elapsed time across calls. Rows show nested recorded operations; horizontal position is not query time.\n"
             "Orange: scheduler methods. Blue: PyTorch and CUDA API operations on the CPU. Gray: thread interval or unrecorded time.\n"
             "GPU labels and green strips show active seconds; dark gray strips show idle time. Overlap does not imply launch attribution.\n"
             "Unrecorded CPU time is not CPU idle time. Python call stacks were not recorded.\n"
             "Open bio3_join_profile.html to zoom and read complete names, small operations, and deeper levels.",
             fontsize=10, linespacing=1.5)
    fig.subplots_adjust(left=0.06, right=0.98, top=0.87, bottom=0.43)
    destination = HERE / "plots/bio3_join_profile"
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BIO-3 CPU and GPU flame graph</title>
<style>
body {font:16px system-ui,sans-serif;color:#222;margin:28px;max-width:1800px}
h1 {font-size:24px} p {max-width:1050px;line-height:1.5}
button {font:inherit;padding:6px 12px;margin-right:10px;cursor:pointer}
#chart {width:100%;overflow:auto} canvas {display:block;cursor:pointer}
#details {white-space:pre-wrap;overflow-wrap:anywhere;font:14px ui-monospace,monospace;min-height:90px}
#selected {overflow-wrap:anywhere;margin:16px 0;font-family:ui-monospace,monospace}
</style>
<h1>BIO-3, pipelined vLLM, CPU intervals and concurrent GPU activity</h1>
<p>Width is <strong>total elapsed seconds across calls</strong>. Children appear below their parent.
Horizontal position is not query time. Click a rectangle to zoom; hover or click to read its complete name.</p>
<p>Orange shows scheduler methods. Blue shows recorded PyTorch and CUDA API operations on the CPU.
Gray marks the full thread interval or time with no recorded CPU operation.
<strong>No recorded CPU operation does not mean CPU idle.</strong> Python work and waiting can occur there;
Python call stacks were not recorded.</p>
<p>Each bottom strip shows <strong>green for time overlapping GPU activity</strong> and dark gray for GPU idle time.
Segments show totals, not event order. GPU work may come from an earlier request, so this is overlap, not launch attribution.</p>
<button id="reset">Reset</button><button id="back">Back</button>
<div id="selected"></div><p id="gpu-summary"></p>
<div id="chart"><canvas id="canvas" aria-label="Interactive CPU flame graph with concurrent GPU activity"></canvas></div>
<pre id="details" role="status"></pre>
<script>
const root = __DATA__;
const colors = __COLORS__;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const details = document.getElementById('details');
let selected = root, history = [], boxes = [];
function color(node) {
  if (node.name.startsWith('vllm.scheduler.')) return colors.orange;
  if (node.name.startsWith('[') || node.name === 'Worker main thread') return colors.gray;
  return colors.blue;
}
function describe(node) {
  details.textContent = node.name + '\nElapsed: ' + node.seconds.toFixed(6) + ' s\n'
    + 'GPU active during these intervals: ' + node.gpu_seconds.toFixed(6) + ' s\n'
    + 'GPU idle during these intervals: ' + node.gpu_idle_seconds.toFixed(6) + ' s\n'
    + 'Elapsed without recorded child operations: ' + node.self_seconds.toFixed(6) + ' s\n'
    + 'Recorded calls: ' + node.calls.toLocaleString();
}
function draw() {
  const width = Math.max(700, document.getElementById('chart').clientWidth);
  boxes = [];
  function collect(node, x, depth) {
    const w = node.seconds / selected.seconds * width;
    if (w < 0.25) return;
    boxes.push({node:node,x:x,y:depth*44,w:w,h:42});
    let childX = x;
    for (const child of node.children) {
      collect(child, childX, depth+1);
      childX += child.seconds / selected.seconds * width;
    }
  }
  collect(selected, 0, 0);
  const height = Math.max(160, ...boxes.map(b => b.y+b.h+35));
  const ratio = window.devicePixelRatio || 1;
  canvas.width = width*ratio; canvas.height = height*ratio;
  canvas.style.width = width+'px'; canvas.style.height = height+'px';
  ctx.scale(ratio,ratio); ctx.font='12px ui-monospace,monospace';
  for (const box of boxes) {
    ctx.fillStyle = color(box.node); ctx.fillRect(box.x,box.y,Math.max(.25,box.w-1),box.h);
    ctx.fillStyle=colors.dark;ctx.fillRect(box.x,box.y+34,Math.max(.25,box.w-1),8);
    const activeWidth=box.w*box.node.gpu_seconds/box.node.seconds;
    ctx.fillStyle=colors.green;ctx.fillRect(box.x,box.y+34,Math.max(0,activeWidth-1),8);
    if (box.w > 30) {
      ctx.save();ctx.beginPath();ctx.rect(box.x+3,box.y,Math.max(0,box.w-6),box.h);ctx.clip();
      ctx.fillStyle='#111';ctx.fillText(box.node.name,box.x+5,box.y+20);ctx.restore();
    }
  }
  ctx.fillStyle='#222';ctx.fillText('0 s',0,height-8);
  const label = selected.seconds.toFixed(3)+' s';
  ctx.fillText(label,width-ctx.measureText(label).width,height-8);
  document.getElementById('selected').textContent=selected.name+' ('+selected.seconds.toFixed(3)+' s)';
  document.getElementById('gpu-summary').textContent='During these intervals: GPU active '
    +selected.gpu_seconds.toFixed(3)+' s; GPU idle '+selected.gpu_idle_seconds.toFixed(3)+' s.';
  document.getElementById('back').disabled=history.length===0;
}
function hit(event) {
  const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
  return boxes.find(b => x>=b.x && x<b.x+b.w && y>=b.y && y<b.y+b.h);
}
canvas.addEventListener('mousemove',event=>{const b=hit(event);if(b)describe(b.node);});
canvas.addEventListener('click',event=>{
  const b=hit(event);if(!b)return;describe(b.node);
  if(b.node!==selected){history.push(selected);selected=b.node;draw();}
});
document.getElementById('reset').onclick=()=>{selected=root;history=[];draw();describe(root);};
document.getElementById('back').onclick=()=>{if(history.length){selected=history.pop();draw();describe(selected);}};
window.addEventListener('resize',draw);draw();describe(root);
</script></html>'''


def write_interactive(root):
    """Save a standalone interactive flame graph."""
    data = json.dumps(root).replace("<", "\\u003c")
    html = HTML.replace("__DATA__", data).replace("__COLORS__", json.dumps({
        "orange": ORANGE, "blue": BLUE, "gray": GRAY, "green": GREEN, "dark": DARK,
    }))
    (HERE / "plots/bio3_join_profile.html").write_text(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    workdir = parser.parse_args().workdir
    result = json.loads((workdir / "result.json").read_text())
    summary = json.loads((workdir / "cpu-flamegraph.json").read_text())
    assert result["query"] == "BIO-3"
    assert summary["source"] == result["result_volume_path"]
    plot_flamegraph(summary["flamegraph"])
    write_interactive(summary["flamegraph"])
