"""Plot the BIO-3 CPU flame graph from saved quail-results summaries.

    W=/tmp/quail-bio3-vllm-profile
    mkdir -p "$W"
    RUN=ablations/vllm-join-profile-20260907T013304Z
    uv run modal volume get quail-results "$RUN/result.json" "$W/result.json"
    uv run modal volume get quail-results "$RUN/cpu-flamegraph.json" "$W/cpu-flamegraph.json"
    uv run modal volume get quail-results "$RUN/gpu-timeline.f64.gz" "$W/gpu-timeline.f64.gz"
    uv run modal volume get quail-results "$RUN/cpu-window.json" "$W/cpu-window.json"
    uv run --with matplotlib python reports/make_bio3_join_profile_plots.py "$W"

The summary aggregates nested CPU intervals from join-0/worker.trace.json.gz
within the join annotation in join-0/driver.trace.json.gz, using
experiments.profile_flamegraph.read_cpu_flamegraph. Names are unchanged.
"""

import argparse
import base64
import gzip
import json
import struct
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch, Rectangle

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
    fig = plt.figure(figsize=(17, max(8, 4.4 + 0.55 * levels)))
    fig.set_layout_engine("none")
    fig.text(0.06, 0.965, "BIO-3, pipelined vLLM, profiled join", fontsize=16)
    fig.text(0.06, 0.90,
             f"GPU idle for {root['gpu_idle_seconds']:.2f} of {root['seconds']:.2f} seconds",
             fontsize=25, weight="bold")
    gpu = fig.add_axes((0.06, 0.70, 0.92, 0.15))
    idle = root["gpu_idle_seconds"]
    active = root["gpu_seconds"]
    gpu.barh(0.5, idle, height=1, left=0, color=DARK)
    gpu.barh(0.5, active, height=1, left=idle, color=GREEN)
    gpu.text(idle / 2, 0.5, f"GPU idle\n{idle:.2f} seconds",
             color="white", fontsize=22, weight="bold", ha="center", va="center")
    gpu.text(idle + active / 2, 0.5, f"GPU active\n{active:.2f} seconds",
             color="black", fontsize=22, weight="bold", ha="center", va="center")
    gpu.set(xlim=(0, root["seconds"]), ylim=(0, 1), xticks=[], yticks=[])
    fig.text(0.06, 0.665,
             "Total time grouped by GPU state, not event order. Profiling overhead is included.", fontsize=11)
    ax = fig.add_axes((0.06, 0.20, 0.92, 0.36))
    total = root["seconds"]
    for start, depth, node in visible:
        patch = Rectangle((start, depth), node["seconds"], 0.92,
                          facecolor=color(node), edgecolor="white", linewidth=0.4)
        ax.add_patch(patch)
        width_fraction = node["seconds"] / total
        if width_fraction > 0.055:
            limit = max(4, int(width_fraction * 170))
            name = node["name"]
            label = name if len(name) <= limit else name[:limit - 3] + "..."
            text = ax.text(start + node["seconds"] / 2, depth + 0.46,
                           f"{label}\n{node['seconds']:.2f} s",
                           ha="center", va="center",
                           fontsize=9, color="black", clip_on=True)
            text.set_clip_path(patch)
    ax.set(xlim=(0, total), ylim=(levels, -0.2), xlabel="seconds", yticks=[])
    ax.set_title("Recorded CPU operations", loc="left", pad=18)
    fig.text(0.06, 0.04,
             "Flame-graph width is total elapsed time across calls. Children appear below their parent. Horizontal position is not query time.\n"
             "Orange: scheduler methods. Blue: PyTorch and CUDA API operations on the CPU. Gray: thread interval or unrecorded time.\n"
             "Unrecorded CPU time is not CPU idle time. Open bio3_join_profile.html to inspect functions and their concurrent GPU activity.",
             fontsize=10, linespacing=1.5)
    destination = HERE / "plots/bio3_join_profile"
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


def load_window(workdir, timeline, source):
    """Combine the saved CPU window with GPU intervals on the same clock."""
    window = json.loads((workdir / "cpu-window.json").read_text())
    assert window["source"] == source
    window["end"] = min(window["end"], window["start"] + 5)
    start, end = window["start"], window["end"]
    window["cpu"] = [[max(start, a), min(end, b), depth, name]
                     for a, b, depth, name in window["cpu"] if b > start and a < end]
    window["gpu"] = [(max(start, a / 1e6), min(end, b / 1e6))
                     for a, b in struct.iter_unpack("<dd", gzip.decompress(timeline))
                     if b / 1e6 > start and a / 1e6 < end]
    return window


def plot_window(window):
    """Save a chronological GPU and CPU view of the selected five seconds."""
    plt.style.use(HERE / "quail.mplstyle")
    start, end = window["start"], window["end"]
    active = sum(b - a for a, b in window["gpu"])
    levels = max(event[2] for event in window["cpu"]) + 1
    fig = plt.figure(figsize=(18, 4.9 + 0.32 * levels))
    fig.set_layout_engine("none")
    fig.text(0.09, 0.965, f"BIO-3, pipelined vLLM, {start:g} to {end:g} seconds after join start",
             fontsize=18, weight="bold")
    fig.text(0.09, 0.92, f"GPU active {active:.2f} seconds; GPU idle {end - start - active:.2f} seconds",
             fontsize=17)
    gpu = fig.add_axes((0.09, 0.69, 0.89, 0.17))
    position = start
    vertices, fills = [], []

    def add(a, b, y, height, fill):
        if b > a:
            vertices.append([(a, y), (b, y), (b, y + height), (a, y + height)])
            fills.append(fill)

    for a, b in window["gpu"]:
        add(position, a, 1, 0.8, DARK)
        add(a, b, 0, 0.8, GREEN)
        position = b
    add(position, end, 1, 0.8, DARK)
    gpu.add_collection(PolyCollection(vertices, facecolors=fills, edgecolors="none"))
    gpu.set(xlim=(start, end), ylim=(1.9, -0.1), yticks=[0.4, 1.4],
            yticklabels=["GPU active", "GPU idle"], xticks=range(int(start), int(end) + 1))
    gpu.tick_params(axis="both", labelsize=12)
    cpu = fig.add_axes((0.09, 0.17, 0.89, 0.40), sharex=gpu)
    cpu.set_title(f"Recorded CPU operations on worker thread {window['thread_id']}", loc="left", fontsize=14)
    vertices, fills = [], []
    for a, b, depth, name in window["cpu"]:
        add(a, b, depth, 0.9, color({"name": name}))
        if b - a > (end - start) * 0.035:
            patch = Rectangle((a, depth), b - a, 0.9, transform=cpu.transData)
            label = cpu.text(a + 0.004, depth + 0.46, name, fontsize=9, va="center", clip_on=True)
            label.set_clip_path(patch)
    cpu.add_collection(PolyCollection(vertices, facecolors=fills, edgecolors="none"))
    schedules = [event for event in window["cpu"] if event[3] == "vllm.scheduler.schedule"]
    if schedules:
        a, b, depth, name = max(schedules, key=lambda event: event[1] - event[0])
        cpu.annotate(name, xy=((a + b) / 2, depth + 0.45),
                     xytext=((a + b) / 2, -0.65), ha="center", fontsize=11,
                     arrowprops={"arrowstyle": "-", "color": DARK, "lw": 0.7})
    cpu.set(xlim=(start, end), ylim=(levels, -0.1), yticks=[], xlabel="seconds")
    cpu.tick_params(axis="x", labelsize=12)
    cpu.xaxis.label.set_size(12)
    fig.legend(handles=[
        Patch(facecolor=BLUE, label="PyTorch / CUDA API calls (CPU)"),
        Patch(facecolor=ORANGE, label="vllm.scheduler.*"),
        Patch(facecolor=GRAY, label="[no recorded CPU operation]"),
    ], loc="lower left", bbox_to_anchor=(0.09, 0.015), ncol=3, fontsize=12,
        borderaxespad=0, handlelength=1.2, handleheight=1.2, columnspacing=3)
    destination = HERE / "plots/bio3_join_window"
    fig.savefig(destination.with_suffix(".png"), dpi=300)
    fig.savefig(destination.with_suffix(".pdf"))
    plt.close(fig)


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>BIO-3 CPU and GPU flame graph</title>
<style>
body {font:16px system-ui,sans-serif;color:#222;margin:28px;max-width:1800px}
h1 {font-size:22px;font-weight:500;margin-bottom:20px} p {max-width:1050px;line-height:1.5}
h2 {font-size:22px;margin:28px 0 14px}
#headline {font-size:clamp(25px,3vw,42px);line-height:1.2;margin:0 0 22px}
.gpu-bar {display:flex;width:100%;height:120px}
.gpu-segment {display:flex;flex-direction:column;align-items:center;justify-content:center;min-width:0;font-size:clamp(16px,2vw,26px);font-weight:650;line-height:1.5}
.gpu-idle {background:__DARK__;color:white}
.gpu-active {background:__GREEN__;color:#111}
.caption {font-size:14px;color:#555;margin:12px 0 30px}
#selection-gpu {margin:16px 0 24px}
#selection-gpu .gpu-bar {height:80px}
#selection-gpu .gpu-segment {font-size:18px}
details {margin:18px 0} summary {cursor:pointer}
input,select {font:inherit;padding:5px}
.timeline-controls {display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:18px 0}
#timeline-start {width:130px}
#timeline-position {width:100%;box-sizing:border-box;margin:12px 0}
#gpu-timeline {width:100%;height:180px;cursor:crosshair}
#timeline-status,#timeline-hover {font:14px ui-monospace,monospace;min-height:22px;overflow-wrap:anywhere}
#cpu-gpu-window {width:100%;cursor:crosshair}
.cpu-legend {display:flex;gap:28px;flex-wrap:wrap;margin:18px 0;font-size:14px}
.cpu-legend span {display:inline-flex;align-items:center;gap:8px}
.cpu-legend i {display:inline-block;width:16px;height:16px}
#window-status,#window-details {font:14px ui-monospace,monospace;white-space:pre-wrap;min-height:40px}
button {font:inherit;padding:6px 12px;margin-right:10px;cursor:pointer}
#chart {width:100%;overflow:auto} canvas {display:block;cursor:pointer}
#details {white-space:pre-wrap;overflow-wrap:anywhere;font:14px ui-monospace,monospace;min-height:90px}
#selected {overflow-wrap:anywhere;margin:16px 0;font-family:ui-monospace,monospace}
</style>
<h1>BIO-3, pipelined vLLM, profiled join</h1>
<h2 id="headline"></h2>
<div id="overview-gpu" class="gpu-bar" role="img" aria-label="Total GPU idle and active time">
  <div class="gpu-segment gpu-idle"></div><div class="gpu-segment gpu-active"></div>
</div>
<p class="caption">Total time grouped by GPU state, not event order. GPU active means at least one recorded kernel or transfer.
Profiling overhead is included.</p>
<h2>Five seconds with GPU and CPU on the same time axis</h2>
<p>480 to 485 seconds. Hover for function names; click to zoom.</p>
<button id="window-reset">Reset to 5 seconds</button>
<a href="bio3_join_window.pdf">Download PDF</a>
<div id="window-status" role="status"></div>
<div class="cpu-legend" aria-label="CPU operation colors">
  <span><i style="background:__BLUE__"></i>PyTorch / CUDA API calls (CPU)</span>
  <span><i style="background:__ORANGE__"></i>vllm.scheduler.*</span>
  <span><i style="background:__GRAY__"></i>[no recorded CPU operation]</span>
</div>
<canvas id="cpu-gpu-window" aria-label="Five seconds of GPU active and idle intervals above nested CPU operations on the same time axis"></canvas>
<pre id="window-details" role="status">Hover a rectangle for its recorded name and interval.</pre>
<h2>GPU active and idle over time</h2>
<p>Each interval uses its recorded start and end time. There is no averaging into percentages.
The initial view shows one second around the first GPU operation.
Zoom in to see brief intervals; some are narrower than a pixel in the whole-join view.</p>
<div class="timeline-controls">
  <label>Window <select id="timeline-window">
    <option value="0.001">1 millisecond</option><option value="0.01">10 milliseconds</option>
    <option value="0.1">100 milliseconds</option><option value="1" selected>1 second</option>
    <option value="10">10 seconds</option><option value="60">60 seconds</option>
    <option value="all">Whole join</option>
  </select></label>
  <label>Start <input id="timeline-start" type="number" min="0" step="0.1" value="0"> seconds</label>
  <button id="timeline-prev">Previous</button><button id="timeline-next">Next</button>
  <button id="timeline-first">First GPU operation</button>
</div>
<div id="timeline-status" role="status">Loading recorded GPU intervals...</div>
<canvas id="gpu-timeline" aria-label="GPU active and idle intervals on a time axis"></canvas>
<input id="timeline-position" type="range" min="0" max="1" step="any" value="0" aria-label="Timeline position">
<div id="timeline-hover" role="status"></div>
<h2>Recorded CPU operations</h2>
<p>Click a function to see its GPU active and idle time. Hover for complete names and call counts.</p>
<details><summary>How to read the CPU flame graph</summary>
<p>Width is <strong>total elapsed seconds across calls</strong>. Children appear below their parent.
Horizontal position is not query time. Click a rectangle to zoom; hover or click to read its complete name.</p>
<p>Orange shows scheduler methods. Blue shows recorded PyTorch and CUDA API operations on the CPU.
Gray marks the full thread interval or time with no recorded CPU operation.
<strong>No recorded CPU operation does not mean CPU idle.</strong> Python work and waiting can occur there;
Python call stacks were not recorded.</p>
<p>GPU work may come from an earlier request, so the GPU breakdown shows overlap, not launch attribution.</p>
</details>
<button id="reset">Reset</button><button id="back">Back</button>
<div id="selected"></div>
<div id="selection-gpu" hidden>
  <p id="gpu-summary"></p>
  <div class="gpu-bar" role="img" aria-label="GPU activity during selected CPU intervals">
    <div class="gpu-segment gpu-idle"></div><div class="gpu-segment gpu-active"></div>
  </div>
</div>
<div id="chart"><canvas id="canvas" aria-label="Interactive CPU flame graph with concurrent GPU activity"></canvas></div>
<pre id="details" role="status"></pre>
<script>
const root = __DATA__;
const colors = __COLORS__;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const details = document.getElementById('details');
let selected = root, history = [], boxes = [];
function gpuBar(element,node) {
  const idle=element.querySelector('.gpu-idle'),active=element.querySelector('.gpu-active');
  idle.style.width=(100*node.gpu_idle_seconds/node.seconds)+'%';
  active.style.width=(100*node.gpu_seconds/node.seconds)+'%';
  idle.textContent='';active.textContent='';
  for (const [target,label,value] of [[idle,'GPU idle',node.gpu_idle_seconds],[active,'GPU active',node.gpu_seconds]]) {
    const name=document.createElement('span'),number=document.createElement('span');
    name.textContent=label;number.textContent=value.toFixed(2)+' seconds';
    target.append(name,number);
    target.style.display=value===0?'none':'flex';
  }
  element.setAttribute('aria-label','GPU idle '+node.gpu_idle_seconds.toFixed(2)+' seconds; GPU active '+node.gpu_seconds.toFixed(2)+' seconds');
}
document.getElementById('headline').textContent='GPU idle for '+root.gpu_idle_seconds.toFixed(2)
  +' of '+root.seconds.toFixed(2)+' seconds';
gpuBar(document.getElementById('overview-gpu'),root);
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
    boxes.push({node:node,x:x,y:depth*34,w:w,h:32});
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
    if (box.w > 30) {
      ctx.save();ctx.beginPath();ctx.rect(box.x+3,box.y,Math.max(0,box.w-6),box.h);ctx.clip();
      ctx.fillStyle='#111';ctx.fillText(box.node.name,box.x+5,box.y+20);ctx.restore();
    }
  }
  ctx.fillStyle='#222';ctx.fillText('0 s',0,height-8);
  const label = selected.seconds.toFixed(3)+' s';
  ctx.fillText(label,width-ctx.measureText(label).width,height-8);
  document.getElementById('selected').textContent=selected.name+' ('+selected.seconds.toFixed(3)+' s)';
  document.getElementById('selection-gpu').hidden=selected===root;
  document.getElementById('gpu-summary').textContent='GPU activity during '+selected.name+' intervals';
  gpuBar(document.querySelector('#selection-gpu .gpu-bar'),selected);
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
const encodedTimeline = '__TIMELINE_DATA__';
__TIMELINE_JS__
const cpuWindow = __CPU_WINDOW__;
__CPU_WINDOW_JS__
</script></html>'''


def write_interactive(root, timeline, window):
    """Save a standalone interactive flame graph."""
    data = json.dumps(root).replace("<", "\\u003c")
    html = HTML.replace("__DARK__", DARK).replace("__GREEN__", GREEN)
    html = html.replace("__BLUE__", BLUE).replace("__ORANGE__", ORANGE).replace("__GRAY__", GRAY)
    html = html.replace("__DATA__", data).replace("__COLORS__", json.dumps({
        "orange": ORANGE, "blue": BLUE, "gray": GRAY, "green": GREEN, "dark": DARK,
    }))
    html = html.replace("__TIMELINE_DATA__", base64.b64encode(timeline).decode("ascii"))
    html = html.replace("__TIMELINE_JS__", (HERE / "gpu_timeline.js").read_text())
    html = html.replace("__CPU_WINDOW__", json.dumps(window).replace("<", "\\u003c"))
    html = html.replace("__CPU_WINDOW_JS__", (HERE / "cpu_gpu_window.js").read_text())
    (HERE / "plots/bio3_join_profile.html").write_text(html)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    workdir = parser.parse_args().workdir
    result = json.loads((workdir / "result.json").read_text())
    summary = json.loads((workdir / "cpu-flamegraph.json").read_text())
    assert result["query"] == "BIO-3"
    assert summary["source"] == result["result_volume_path"]
    timeline = (workdir / "gpu-timeline.f64.gz").read_bytes()
    window = load_window(workdir, timeline, summary["source"])
    plot_flamegraph(summary["flamegraph"])
    plot_window(window)
    write_interactive(summary["flamegraph"], timeline, window)
