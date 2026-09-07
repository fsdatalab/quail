const windowCanvas = document.getElementById('cpu-gpu-window');
const windowContext = windowCanvas.getContext('2d');
const windowDetails = document.getElementById('window-details');
let viewStart = cpuWindow.start, viewEnd = cpuWindow.end, windowBoxes = [];

function windowColor(name) {
  return name.startsWith('[') ? colors.gray : name.startsWith('vllm.scheduler.') ? colors.orange : colors.blue;
}

function drawCpuGpuWindow() {
  const width = Math.max(350, windowCanvas.clientWidth), left = 115, right = width - 16;
  const depth = cpuWindow.cpu.reduce((maximum, event) => Math.max(maximum, event[2]), 0);
  const cpuTop = 183, rowHeight = 27, height = cpuTop + (depth + 1) * rowHeight + 55;
  const ratio = window.devicePixelRatio || 1, scale = (right - left) / (viewEnd - viewStart);
  windowCanvas.style.height = height + 'px';
  windowCanvas.width = width * ratio; windowCanvas.height = height * ratio;
  windowContext.scale(ratio, ratio);
  windowBoxes = [];
  function rectangle(start, end, y, h, fill, name) {
    const a = Math.max(viewStart, start), b = Math.min(viewEnd, end);
    if (b <= a) return;
    const x = left + (a - viewStart) * scale, w = (b - a) * scale;
    windowContext.fillStyle = fill;
    windowContext.fillRect(x, y, w, h);
    windowBoxes.push({x, y, w, h, start, end, name});
    if (y >= cpuTop && w > 55) {
      windowContext.save();
      windowContext.beginPath(); windowContext.rect(x + 2, y, Math.max(0, w - 4), h); windowContext.clip();
      windowContext.font = '12px ui-monospace,monospace'; windowContext.fillStyle = '#111';
      windowContext.fillText(name, x + 4, y + 17);
      windowContext.restore();
    }
  }
  let position = cpuWindow.start, active = 0;
  for (const [start, end] of cpuWindow.gpu) {
    rectangle(position, start, 77, 42, colors.dark, 'GPU idle');
    rectangle(start, end, 20, 42, colors.green, 'GPU active');
    active += Math.max(0, Math.min(end, viewEnd) - Math.max(start, viewStart));
    position = end;
  }
  rectangle(position, cpuWindow.end, 77, 42, colors.dark, 'GPU idle');
  for (const [start, end, level, name] of cpuWindow.cpu) {
    rectangle(start, end, cpuTop + level * rowHeight, rowHeight - 2, windowColor(name), name);
  }
  windowContext.fillStyle = '#222'; windowContext.font = '14px system-ui,sans-serif';
  windowContext.fillText('GPU active', 0, 46);
  windowContext.fillText('GPU idle', 0, 103);
  windowContext.fillText('CPU', 0, cpuTop + 18);
  windowContext.fillText('Worker thread ' + cpuWindow.thread_id, left, cpuTop - 16);
  const decimals = viewEnd - viewStart < 1 ? 6 : 1;
  windowContext.font = '12px ui-monospace,monospace';
  for (let i = 0; i <= 5; i++) {
    const x = left + (right - left) * i / 5, t = viewStart + (viewEnd - viewStart) * i / 5;
    windowContext.textAlign = i === 0 ? 'left' : i === 5 ? 'right' : 'center';
    windowContext.fillText(t.toFixed(decimals), x, 143);
    windowContext.fillText(t.toFixed(decimals), x, height - 30);
  }
  windowContext.textAlign = 'center';
  windowContext.fillText('seconds after join start', (left + right) / 2, height - 8);
  windowContext.textAlign = 'left';
  document.getElementById('window-status').textContent = viewStart.toFixed(6) + ' to '
    + viewEnd.toFixed(6) + ' seconds. GPU active ' + active.toFixed(6)
    + ' s; GPU idle ' + (viewEnd - viewStart - active).toFixed(6) + ' s.';
}

function windowHit(event) {
  const bounds = windowCanvas.getBoundingClientRect();
  const x = event.clientX - bounds.left, y = event.clientY - bounds.top;
  return windowBoxes.find(box => x >= box.x && x < box.x + box.w && y >= box.y && y < box.y + box.h);
}
windowCanvas.addEventListener('mousemove', event => {
  const box = windowHit(event);
  windowDetails.textContent = box ? box.name + '\n' + box.start.toFixed(9) + ' to '
    + box.end.toFixed(9) + ' seconds (' + ((box.end - box.start) * 1000).toFixed(6) + ' milliseconds)'
    : 'Hover a rectangle for its recorded name and interval.';
});
windowCanvas.addEventListener('click', event => {
  const box = windowHit(event);
  if (!box) return;
  const padding = Math.max(0.00001, (box.end - box.start) * 0.15);
  viewStart = Math.max(cpuWindow.start, box.start - padding);
  viewEnd = Math.min(cpuWindow.end, box.end + padding);
  drawCpuGpuWindow();
});
document.getElementById('window-reset').onclick = () => {
  viewStart = cpuWindow.start; viewEnd = cpuWindow.end; drawCpuGpuWindow();
};
window.addEventListener('resize', drawCpuGpuWindow);
drawCpuGpuWindow();
