const gpuCanvas = document.getElementById('gpu-timeline');
const gpuContext = gpuCanvas.getContext('2d');
const timelineStatus = document.getElementById('timeline-status');
const timelineStartInput = document.getElementById('timeline-start');
const timelineWindowInput = document.getElementById('timeline-window');
const timelinePositionInput = document.getElementById('timeline-position');
let gpuEndpoints = null, timelineStart = 0, timelineSpan = 1, timelineGeometry = null;

function firstGpuIntervalEndingAfter(time) {
  let low = 0, high = gpuEndpoints.length / 2;
  while (low < high) {
    const middle = Math.floor((low + high) / 2);
    if (gpuEndpoints[2 * middle + 1] <= time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function drawGpuTimeline() {
  if (!gpuEndpoints) return;
  timelineSpan = Math.min(root.seconds, Math.max(0.001, timelineSpan));
  timelineStart = Math.min(root.seconds - timelineSpan, Math.max(0, timelineStart));
  const end = timelineStart + timelineSpan;
  const width = Math.max(350, gpuCanvas.clientWidth), height = 180, left = 115, right = width - 16;
  const ratio = window.devicePixelRatio || 1;
  gpuCanvas.width = width * ratio; gpuCanvas.height = height * ratio;
  gpuContext.scale(ratio, ratio);
  gpuContext.font = '14px system-ui,sans-serif';
  gpuContext.fillStyle = '#222';
  gpuContext.fillText('GPU active', 0, 52);
  gpuContext.fillText('GPU idle', 0, 110);
  const scale = (right - left) / timelineSpan;
  timelineGeometry = {left, right, scale};
  function span(start, stop, active) {
    if (stop <= start) return;
    gpuContext.fillStyle = active ? colors.green : colors.dark;
    gpuContext.fillRect(left + (start - timelineStart) * scale, active ? 22 : 80,
      (stop - start) * scale, 46);
  }
  let position = timelineStart, activeSeconds = 0;
  for (let i = firstGpuIntervalEndingAfter(timelineStart); i < gpuEndpoints.length / 2; i++) {
    const start = Math.max(timelineStart, gpuEndpoints[2 * i]);
    const stop = Math.min(end, gpuEndpoints[2 * i + 1]);
    if (start >= end) break;
    span(position, start, false);
    span(start, stop, true);
    activeSeconds += stop - start;
    position = stop;
  }
  span(position, end, false);
  const decimals = timelineSpan < 1 ? 6 : 3;
  gpuContext.fillStyle = '#333';
  gpuContext.font = '12px ui-monospace,monospace';
  for (let i = 0; i <= 4; i++) {
    const x = left + (right - left) * i / 4;
    const label = (timelineStart + timelineSpan * i / 4).toFixed(decimals);
    gpuContext.textAlign = i === 0 ? 'left' : i === 4 ? 'right' : 'center';
    gpuContext.fillText(label, x, 151);
  }
  gpuContext.textAlign = 'center';
  gpuContext.fillText('seconds after join start', (left + right) / 2, 173);
  gpuContext.textAlign = 'left';
  timelineStartInput.value = timelineStart.toFixed(6);
  timelineStartInput.max = Math.max(0, root.seconds - timelineSpan);
  timelineStartInput.step = timelineSpan / 10;
  timelinePositionInput.max = Math.max(0, root.seconds - timelineSpan);
  timelinePositionInput.value = timelineStart;
  document.getElementById('timeline-prev').disabled = timelineStart === 0;
  document.getElementById('timeline-next').disabled = end >= root.seconds;
  timelineStatus.textContent = 'Showing ' + timelineStart.toFixed(6) + ' to ' + end.toFixed(6)
    + ' seconds. In this window: GPU active ' + activeSeconds.toFixed(6)
    + ' s; GPU idle ' + Math.max(0, timelineSpan - activeSeconds).toFixed(6) + ' s.';
}

function firstGpuOperation() {
  if (!gpuEndpoints) return;
  timelineStart = gpuEndpoints.length ? gpuEndpoints[0] - timelineSpan * 0.2 : 0;
  drawGpuTimeline();
}

timelineWindowInput.onchange = () => {
  const center = timelineStart + timelineSpan / 2;
  timelineSpan = timelineWindowInput.value === 'all' ? root.seconds : Number(timelineWindowInput.value);
  timelineStart = center - timelineSpan / 2;
  drawGpuTimeline();
};
timelineStartInput.onchange = () => {
  const value = Number(timelineStartInput.value);
  if (Number.isFinite(value)) timelineStart = value;
  drawGpuTimeline();
};
timelinePositionInput.oninput = () => {timelineStart = Number(timelinePositionInput.value); drawGpuTimeline();};
document.getElementById('timeline-prev').onclick = () => {timelineStart -= timelineSpan; drawGpuTimeline();};
document.getElementById('timeline-next').onclick = () => {timelineStart += timelineSpan; drawGpuTimeline();};
document.getElementById('timeline-first').onclick = firstGpuOperation;
gpuCanvas.addEventListener('mousemove', event => {
  if (!gpuEndpoints || !timelineGeometry) return;
  const x = event.clientX - gpuCanvas.getBoundingClientRect().left;
  if (x < timelineGeometry.left || x > timelineGeometry.right) return;
  const time = timelineStart + (x - timelineGeometry.left) / timelineGeometry.scale;
  const i = firstGpuIntervalEndingAfter(time), count = gpuEndpoints.length / 2;
  const active = i < count && gpuEndpoints[2 * i] <= time;
  const start = active ? gpuEndpoints[2 * i] : i > 0 ? gpuEndpoints[2 * i - 1] : 0;
  const end = active ? gpuEndpoints[2 * i + 1] : i < count ? gpuEndpoints[2 * i] : root.seconds;
  document.getElementById('timeline-hover').textContent = 'GPU ' + (active ? 'active' : 'idle')
    + ' from ' + start.toFixed(9) + ' to ' + end.toFixed(9) + ' seconds ('
    + ((end - start) * 1e6).toFixed(3) + ' microseconds).';
});
window.addEventListener('resize', drawGpuTimeline);

async function loadGpuTimeline() {
  const compressed = Uint8Array.from(atob(encodedTimeline), value => value.charCodeAt(0));
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream('gzip'));
  const buffer = await new Response(stream).arrayBuffer();
  if (buffer.byteLength % 16 !== 0) throw new Error('Invalid GPU interval encoding');
  const view = new DataView(buffer);
  gpuEndpoints = new Float64Array(buffer.byteLength / 8);
  let activeSeconds = 0, previous = 0;
  for (let i = 0; i < gpuEndpoints.length; i += 2) {
    const start = view.getFloat64(i * 8, true) / 1e6;
    const end = view.getFloat64((i + 1) * 8, true) / 1e6;
    if (start < previous || end <= start || end > root.seconds + 1e-8) throw new Error('Invalid GPU interval');
    gpuEndpoints[i] = start; gpuEndpoints[i + 1] = end;
    activeSeconds += end - start; previous = end;
  }
  if (Math.abs(activeSeconds - root.gpu_seconds) > 1e-6) throw new Error('GPU timeline total differs from flame graph');
  firstGpuOperation();
}
loadGpuTimeline().catch(error => {timelineStatus.textContent = 'Could not load GPU timeline: ' + error.message;});
