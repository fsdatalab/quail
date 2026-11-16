// The share of menus that sell drinks and that mention a war, by year.
// The rows are the query's result in content/docs/cookbook/menus.mdx; the
// SVG is static, so the page prerenders it.

const ROWS = [
  // year, menus, drinks, war
  [1914, 59, 0.356, 0.017],
  [1915, 59, 0.186, 0.034],
  [1916, 60, 0.15, 0],
  [1917, 60, 0.067, 0.117],
  [1918, 60, 0.133, 0.567],
  [1919, 59, 0.034, 0.051],
  [1920, 59, 0.085, 0.034],
  [1921, 60, 0.033, 0.017],
  [1926, 30, 0.033, 0],
  [1928, 49, 0.143, 0],
  [1931, 23, 0.043, 0],
  [1932, 53, 0.057, 0],
  [1933, 60, 0.3, 0],
  [1934, 36, 0.111, 0],
  [1935, 57, 0.211, 0.018],
  [1937, 56, 0.214, 0],
  [1938, 55, 0.091, 0.018],
  [1939, 54, 0.222, 0],
  [1940, 53, 0.321, 0.038],
  [1941, 44, 0.25, 0.023],
  [1943, 56, 0.179, 0.179],
  [1944, 56, 0.268, 0.179],
  [1945, 60, 0.433, 0.383],
  [1946, 59, 0.153, 0.237],
  [1947, 43, 0.233, 0.023],
  [1948, 58, 0.224, 0.017],
  [1949, 59, 0.271, 0.017],
] as const;

const FIRST_YEAR = 1914;
const LAST_YEAR = 1949;
const LEFT = 48;
const RIGHT = 640;
const TOP = 34;
const BOTTOM = 250;
const MAX_SHARE = 0.6;

function x(year: number) {
  return LEFT + ((year - FIRST_YEAR) / (LAST_YEAR - FIRST_YEAR)) * (RIGHT - LEFT);
}

function y(share: number) {
  return BOTTOM - (share / MAX_SHARE) * (BOTTOM - TOP);
}

// one path per run of consecutive years, so a gap shows as a gap
function path(index: 2 | 3) {
  let found = '';
  ROWS.forEach((row, position) => {
    const previous = ROWS[position - 1];
    const joined = previous && row[0] - previous[0] === 1;
    found += `${joined ? 'L' : 'M'}${x(row[0]).toFixed(1)} ${y(row[index]).toFixed(1)} `;
  });
  return found.trim();
}

function points(index: 2 | 3, kind: string, name: string) {
  return ROWS.map((row) => `<circle cx="${x(row[0]).toFixed(1)}" cy="${y(row[index]).toFixed(1)}" r="4" class="${kind}"><title>${row[0]}: ${Math.round(row[index] * 100)}% of ${row[1]} menus ${name}</title></circle>`).join('');
}

const GRID = [0, 0.2, 0.4, 0.6].map((share) =>
  `<line x1="${LEFT}" y1="${y(share)}" x2="${RIGHT}" y2="${y(share)}" class="grid"/>
<text x="${LEFT - 8}" y="${y(share) + 4}" text-anchor="end" class="tick">${share * 100}%</text>`).join('');

const YEARS = [1915, 1920, 1925, 1930, 1935, 1940, 1945].map((year) =>
  `<text x="${x(year)}" y="${BOTTOM + 20}" text-anchor="middle" class="tick">${year}</text>`).join('');

function band(start: number, end: number, label: string) {
  return `<rect x="${x(start)}" y="${TOP}" width="${x(end) - x(start)}" height="${BOTTOM - TOP}" class="band"/>
<text x="${(x(start) + x(end)) / 2}" y="${TOP - 10}" text-anchor="middle" class="band-label">${label}</text>`;
}

const SVG = `${band(1917.3, 1918.9, 'World War I')}
${band(1920, 1933.95, 'Prohibition')}
${band(1941.95, 1945.7, 'World War II')}
${GRID}
<line x1="${LEFT}" y1="${BOTTOM}" x2="${RIGHT}" y2="${BOTTOM}" class="axis"/>
${YEARS}
<path d="${path(2)}" class="line drinks"/>
<path d="${path(3)}" class="line war"/>
${points(2, 'drinks', 'sell drinks')}
${points(3, 'war', 'mention a war')}
<text x="${x(1940) - 8}" y="${y(0.321) - 8}" text-anchor="end" class="label">sell drinks</text>
<text x="${x(1918) + 8}" y="${y(0.567) + 4}" class="label">mention a war</text>`;

export function MenuHistoryChart() {
  return (
    <figure className="menu-history not-prose">
      <figcaption>Share of menus, by year</figcaption>
      <div className="legend" aria-hidden="true">
        <span><i className="drinks" />sell beer, wine, or liquor</span>
        <span><i className="war" />mention a war, rationing, or a meatless or wheatless day</span>
      </div>
      <div className="plot">
        <svg
          viewBox="0 0 660 280"
          role="img"
          aria-label="Line chart, 1914 to 1949: menus selling drinks fall from 36% in 1914 to 3–9% from 1919 to 1932, then return to 20–43%; menus mentioning a war rise to 57% in 1918 and to 18–38% from 1943 to 1946."
          dangerouslySetInnerHTML={{ __html: SVG }}
        />
      </div>
    </figure>
  );
}
