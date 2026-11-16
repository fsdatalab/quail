'use client';

// The share of menus that sell drinks and that mention a war, by year: the
// result of the query in content/docs/cookbook/menus.mdx.

import { useState, type PointerEvent } from 'react';

type Row = { year: number; menus: number; drinks: number; war: number };

const ROWS: Row[] = [
  [1914, 59, 0.356, 0.017], [1915, 59, 0.186, 0.034], [1916, 60, 0.15, 0],
  [1917, 60, 0.067, 0.117], [1918, 60, 0.133, 0.567], [1919, 59, 0.034, 0.051],
  [1920, 59, 0.085, 0.034], [1921, 60, 0.033, 0.017], [1926, 30, 0.033, 0],
  [1928, 49, 0.143, 0], [1931, 23, 0.043, 0], [1932, 53, 0.057, 0],
  [1933, 60, 0.3, 0], [1934, 36, 0.111, 0], [1935, 57, 0.211, 0.018],
  [1937, 56, 0.214, 0], [1938, 55, 0.091, 0.018], [1939, 54, 0.222, 0],
  [1940, 53, 0.321, 0.038], [1941, 44, 0.25, 0.023], [1943, 56, 0.179, 0.179],
  [1944, 56, 0.268, 0.179], [1945, 60, 0.433, 0.383], [1946, 59, 0.153, 0.237],
  [1947, 43, 0.233, 0.023], [1948, 58, 0.224, 0.017], [1949, 59, 0.271, 0.017],
].map(([year, menus, drinks, war]) => ({ year, menus, drinks, war }));

const SERIES = [
  { key: 'drinks', label: 'sell beer, wine, or liquor', short: 'sell drinks' },
  { key: 'war', label: 'mention a war, rationing, or a meatless or wheatless day', short: 'mention a war' },
] as const;
type SeriesKey = (typeof SERIES)[number]['key'];

const EVENTS = [
  { start: 1917.3, end: 1918.9, label: 'World War I' },
  { start: 1920, end: 1933.95, label: 'Prohibition' },
  { start: 1941.95, end: 1945.7, label: 'World War II' },
];

const WIDTH = 660;
const HEIGHT = 280;
const LEFT = 48;
const RIGHT = 640;
const TOP = 34;
const BOTTOM = 250;
const FIRST_YEAR = 1914;
const LAST_YEAR = 1949;
const MAX_SHARE = 0.6;

const x = (year: number) =>
  LEFT + ((year - FIRST_YEAR) / (LAST_YEAR - FIRST_YEAR)) * (RIGHT - LEFT);
const y = (share: number) => BOTTOM - (share / MAX_SHARE) * (BOTTOM - TOP);
const percent = (share: number) => `${Math.round(share * 100)}%`;

// one path per run of consecutive years, so a missing year shows as a gap
function path(key: SeriesKey) {
  return ROWS.map((row, index) => {
    const joined = index > 0 && row.year - ROWS[index - 1].year === 1;
    return `${joined ? 'L' : 'M'}${x(row.year).toFixed(1)} ${y(row[key]).toFixed(1)}`;
  }).join(' ');
}

export function MenuHistoryChart() {
  const [hidden, setHidden] = useState<Set<SeriesKey>>(new Set());
  const [active, setActive] = useState<Row | null>(null);
  const shown = SERIES.filter((series) => !hidden.has(series.key));

  function toggle(key: SeriesKey) {
    setHidden((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function track(event: PointerEvent<SVGSVGElement>) {
    const box = event.currentTarget.getBoundingClientRect();
    const year = FIRST_YEAR + ((event.clientX - box.left) / box.width * WIDTH - LEFT)
      / (RIGHT - LEFT) * (LAST_YEAR - FIRST_YEAR);
    const nearest = ROWS.reduce((best, row) =>
      Math.abs(row.year - year) < Math.abs(best.year - year) ? row : best);
    setActive(Math.abs(nearest.year - year) <= 1.5 ? nearest : null);
  }

  return (
    <figure className="menu-history not-prose">
      <figcaption>Share of menus, by year</figcaption>
      <div className="legend">
        {SERIES.map((series) => (
          <button
            key={series.key}
            type="button"
            aria-pressed={!hidden.has(series.key)}
            onClick={() => toggle(series.key)}
          >
            <i className={series.key} />
            {series.label}
          </button>
        ))}
      </div>
      <div className="plot">
        <svg
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label="Line chart, 1914 to 1949: menus selling drinks fall from 36% in 1914 to 3–9% from 1919 to 1932, then return to 20–43%; menus mentioning a war rise to 57% in 1918 and to 18–38% from 1943 to 1946."
          onPointerMove={track}
          onPointerLeave={() => setActive(null)}
        >
          {EVENTS.map((event) => (
            <g key={event.label}>
              <rect x={x(event.start)} y={TOP} width={x(event.end) - x(event.start)}
                    height={BOTTOM - TOP} className="band" />
              <text x={(x(event.start) + x(event.end)) / 2} y={TOP - 10}
                    textAnchor="middle" className="band-label">{event.label}</text>
            </g>
          ))}
          {[0, 0.2, 0.4, 0.6].map((share) => (
            <g key={share}>
              <line x1={LEFT} y1={y(share)} x2={RIGHT} y2={y(share)} className="grid" />
              <text x={LEFT - 8} y={y(share) + 4} textAnchor="end" className="tick">
                {percent(share)}
              </text>
            </g>
          ))}
          <line x1={LEFT} y1={BOTTOM} x2={RIGHT} y2={BOTTOM} className="axis" />
          {[1915, 1920, 1925, 1930, 1935, 1940, 1945].map((year) => (
            <text key={year} x={x(year)} y={BOTTOM + 20} textAnchor="middle" className="tick">
              {year}
            </text>
          ))}
          {active ? (
            <line x1={x(active.year)} y1={TOP} x2={x(active.year)} y2={BOTTOM}
                  className="crosshair" />
          ) : null}
          {shown.map((series) => (
            <g key={series.key}>
              <path d={path(series.key)} className={`line ${series.key}`} />
              {ROWS.map((row) => (
                <circle key={row.year} cx={x(row.year)} cy={y(row[series.key])}
                        r={active?.year === row.year ? 6 : 4} className={series.key} />
              ))}
            </g>
          ))}
          {!active && !hidden.has('drinks') ? (
            <text x={x(1940) - 8} y={y(0.321) - 8} textAnchor="end" className="label">
              sell drinks
            </text>
          ) : null}
          {!active && !hidden.has('war') ? (
            <text x={x(1918) + 8} y={y(0.567) + 4} className="label">mention a war</text>
          ) : null}
        </svg>
        {active ? (
          <div
            className="tooltip"
            style={{ left: `${(x(active.year) / WIDTH) * 100}%` }}
            data-side={active.year > 1938 ? 'left' : 'right'}
          >
            <strong>{active.year}</strong>
            <span className="muted">{active.menus} menus</span>
            {shown.map((series) => (
              <span key={series.key}>
                <i className={series.key} />
                {series.short} <b>{percent(active[series.key])}</b>
              </span>
            ))}
          </div>
        ) : null}
      </div>
    </figure>
  );
}

export function MenuHistoryTable() {
  return (
    <div className="menu-table not-prose">
      <table>
        <thead>
          <tr>
            <th>year</th>
            <th>menus</th>
            <th>drinks</th>
            <th>war</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map((row) => (
            <tr key={row.year}>
              <td>{row.year}</td>
              <td>{row.menus}</td>
              <td>{row.drinks}</td>
              <td>{row.war}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
