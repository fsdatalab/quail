// How a query moves between DataFusion and Quail, drawn as two lanes.
// The SVG is static, so the page prerenders it.

const ROWS = [
  // y, DataFusion title and line, Quail title and line, arrow right, arrow left
  [52, 'Parser and analyzer', 'SQL text → logical plan',
    'lower_ai_calls', 'each AI call becomes an AiNode',
    'logical plan with AI function calls', 'plan with AiNodes'],
  [126, 'Logical optimizer', 'pushes plain filters, limits, columns',
    'Quail’s optimizer rules', 'merge calls, set options, place AiNodes',
    'whole logical plan', 'same plan, AiNode options set'],
  [200, 'Query planner', 'calls Quail before physical planning',
    'claim_model_regions', 'adjacent AiNodes become one region',
    'optimized logical plan', 'plan with ModelRegion nodes'],
  [274, 'Physical planner', 'scans, hash joins, aggregates, sorts',
    'Extension planner', 'each region becomes a ModelRegionExec',
    'a ModelRegion and its physical inputs', 'one ModelRegionExec, not yet planned'],
] as const;

function box(x: number, y: number, kind: string, title: string, line: string) {
  return `<rect x="${x}" y="${y}" width="250" height="56" rx="5" class="${kind}"/>
<text x="${x + 14}" y="${y + 22}" class="t">${title}</text>
<text x="${x + 14}" y="${y + 42}" class="d">${line}</text>`;
}

function exchange(y: number, right: string, left: string) {
  return `<line x1="260" y1="${y + 20}" x2="509" y2="${y + 20}" class="ln" marker-end="url(#planning-arrow)"/>
<text x="385" y="${y + 15}" text-anchor="middle" class="e">${right}</text>
<line x1="510" y1="${y + 42}" x2="261" y2="${y + 42}" class="ln" marker-end="url(#planning-arrow)"/>
<text x="385" y="${y + 37}" text-anchor="middle" class="e">${left}</text>`;
}

const PLAN_TIME = ROWS.map(([y, title, line, quailTitle, quailLine, right, left], index) =>
  box(10, y, 'b-df', title, line) + box(510, y, 'b-q', quailTitle, quailLine)
  + exchange(y, right, left)
  + (index < ROWS.length - 1
    ? `<line x1="135" y1="${y + 56}" x2="135" y2="${y + 73}" class="ln" marker-end="url(#planning-arrow)"/>`
    : '')).join('');

const SVG = `<defs><marker id="planning-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="currentColor"/></marker></defs>
<text x="135" y="20" text-anchor="middle" class="lane-df">APACHE DATAFUSION</text>
<text x="635" y="20" text-anchor="middle" class="lane-q">QUAIL</text>
<text x="10" y="42" class="ph">PLAN TIME <tspan class="phs">no rows are read</tspan></text>
${PLAN_TIME}
<line x1="10" y1="346" x2="490" y2="346" class="div"/>
<text x="10" y="366" class="ph">RUN TIME <tspan class="phs">rows flow as Arrow batches</tspan></text>
<line x1="635" y1="330" x2="635" y2="373" class="ln" marker-end="url(#planning-arrow)"/>
<text x="645" y="352" class="e">AiNodes</text>
<rect x="500" y="366" width="266" height="182" rx="7" class="b-region"/>
<text x="635" y="540" text-anchor="middle" class="r">MODELREGIONEXEC</text>
${box(10, 374, 'b-df', 'Operators below the region', 'scans, plain filters, plain joins')}
${box(510, 374, 'b-q', 'Region planner', 'orders the operators by GPU time')}
<line x1="260" y1="402" x2="509" y2="402" class="ln" marker-end="url(#planning-arrow)"/>
<text x="385" y="397" text-anchor="middle" class="e">every input row of the region</text>
<line x1="635" y1="430" x2="635" y2="461" class="ln" marker-end="url(#planning-arrow)"/>
<text x="645" y="450" class="e">region plan</text>
${box(10, 462, 'b-df', 'Operators above the region', 'plain joins, aggregates, sorts, limits')}
${box(510, 462, 'b-q', 'Executor', 'runs the region on GPU workers')}
<line x1="510" y1="482" x2="261" y2="482" class="ln" marker-end="url(#planning-arrow)"/>
<text x="385" y="477" text-anchor="middle" class="e">finished rows, as they complete</text>
<line x1="260" y1="504" x2="509" y2="504" class="ln" marker-end="url(#planning-arrow)"/>
<text x="385" y="499" text-anchor="middle" class="e">dynamic filters from joins and sorts</text>
<line x1="135" y1="518" x2="135" y2="550" class="ln" marker-end="url(#planning-arrow)"/>
<text x="145" y="542" class="e">query result</text>`;

export function PlanningFigure() {
  return (
    <figure className="planning-figure not-prose">
      <div className="scroll">
        <svg
          viewBox="0 0 776 560"
          role="img"
          aria-label="At plan time DataFusion hands Quail the logical plan four times and gets a plan back. At run time DataFusion streams rows into each ModelRegionExec, where Quail plans the region and runs it on GPU workers, and finished rows return to DataFusion."
          dangerouslySetInnerHTML={{ __html: SVG }}
        />
      </div>
      <figcaption>
        Each arrow is labeled with what is handed over. Quail sees no rows
        until run time, so Quail plans each model region after the region’s
        input rows arrive.
      </figcaption>
    </figure>
  );
}
