// The share of menus that sell drinks and that mention a war, by year.
// The rows are the query's result in "A century of menus"; the SVG is
// static, so the page prerenders it.

const ROWS = [
  // year, menus, drinks, war
  [1914, 99, 0.293, 0.01],
  [1915, 100, 0.22, 0.03],
  [1916, 99, 0.111, 0.01],
  [1917, 100, 0.12, 0.21],
  [1918, 100, 0.1, 0.57],
  [1919, 99, 0.04, 0.051],
  [1920, 96, 0.063, 0.042],
  [1921, 100, 0.04, 0.02],
  [1926, 30, 0.033, 0],
  [1928, 49, 0.143, 0],
  [1931, 23, 0.043, 0],
  [1932, 53, 0.057, 0],
  [1933, 100, 0.36, 0],
  [1934, 36, 0.111, 0],
  [1935, 74, 0.216, 0.014],
  [1937, 93, 0.215, 0],
] as const;

const FIRST_YEAR = 1914;
const LAST_YEAR = 1937;
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

const YEARS = [1915, 1920, 1925, 1930, 1935].map((year) =>
  `<text x="${x(year)}" y="${BOTTOM + 20}" text-anchor="middle" class="tick">${year}</text>`).join('');

const SVG = `<rect x="${x(1920)}" y="${TOP}" width="${x(1933.95) - x(1920)}" height="${BOTTOM - TOP}" class="band"/>
<text x="${(x(1920) + x(1933.95)) / 2}" y="${TOP - 10}" text-anchor="middle" class="band-label">Prohibition, 1920–1933</text>
<rect x="${x(1917.3)}" y="${TOP}" width="${x(1918.9) - x(1917.3)}" height="${BOTTOM - TOP}" class="band"/>
<text x="${(x(1917.3) + x(1918.9)) / 2}" y="${TOP - 10}" text-anchor="middle" class="band-label">US at war</text>
${GRID}
<line x1="${LEFT}" y1="${BOTTOM}" x2="${RIGHT}" y2="${BOTTOM}" class="axis"/>
${YEARS}
<path d="${path(2)}" class="line drinks"/>
<path d="${path(3)}" class="line war"/>
${points(2, 'drinks', 'sell drinks')}
${points(3, 'war', 'mention a war')}
<text x="${x(1933) + 8}" y="${y(0.36) - 6}" class="label drinks-label">sell drinks</text>
<text x="${x(1918) + 8}" y="${y(0.57) + 4}" class="label war-label">mention a war</text>`;

export function MenuHistoryChart() {
  return (
    <figure className="menu-history not-prose">
      <figcaption>Share of menus, by year</figcaption>
      <div className="legend" aria-hidden="true">
        <span><i className="drinks" />sell beer, wine, or liquor</span>
        <span><i className="war" />mention a war, rationing, or a meatless day</span>
      </div>
      <div className="plot">
        <svg
          viewBox="0 0 660 280"
          role="img"
          aria-label="Line chart: menus selling drinks fall from 29% in 1914 to 4–6% from 1919 to 1932, then rise to 36% in 1933; menus mentioning a war rise to 21% in 1917 and 57% in 1918."
          dangerouslySetInnerHTML={{ __html: SVG }}
        />
      </div>
    </figure>
  );
}
