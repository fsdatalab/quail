"""Report the model answers Quail asks that a query's rows do not need.

    uv run python tools/needed_answers.py --seeds 0,1,4,7,40 --queries 40
    uv run python tools/needed_answers.py --seeds 0 --queries 12 --sufficient

For each generated query (``tests/helpers/generated.py``) it counts the
answers a run with no pushdown asks, the answers one change shows
needed, and the answers Quail asks; with ``--sufficient``, also a set
of answers that together decide the rows (``tests/helpers/needed.py``).
It prints the totals by query shape, then each query that asked an
answer it did not need.
"""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from helpers.generated import Queries, ai_sql  # noqa: E402
from helpers.needed import NeededAnswers  # noqa: E402

SHAPES = (
    (" ORDER BY ", "ORDER BY and LIMIT"),
    ("UNION ALL", "set operation"), ("INTERSECT", "set operation"),
    ("EXCEPT", "set operation"),
    ("GROUP BY", "GROUP BY"),
    ("SELECT DISTINCT", "DISTINCT"),
    ("SELECT COUNT(*)", "scalar subquery"),
    ("EXISTS", "EXISTS or IN"), (" IN (SELECT", "EXISTS or IN"),
    ("SEMI JOIN", "semi or anti join"), ("ANTI JOIN", "semi or anti join"),
    (" JOIN T ON", "AI join"),
    (" JOIN V ", "join on an answer"),
    (" OR ", "one table, OR"),
    ("CASE WHEN", "one table, CASE"),
)


def shape(sql: str) -> str:
    """Return the name of a generated query's shape."""
    upper = sql.upper()
    return next((name for text, name in SHAPES if text in upper), "one table")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--queries", type=int, default=12)
    parser.add_argument("--sufficient", action="store_true")
    options = parser.parse_args()
    totals = defaultdict(Counter)
    wasteful = []
    for seed in (int(seed) for seed in options.seeds.split(",")):
        runs = NeededAnswers(seed)
        queries = Queries(seed)
        try:
            for _ in range(options.queries):
                found = runs.asked(ai_sql(queries.query()))
                counts = totals[shape(found.sql)]
                counts.update(queries=1, plain=len(found.every),
                              needed=len(found.needed), asked=len(found.asked),
                              unneeded=len(found.unneeded),
                              missed=len(found.missed))
                if options.sufficient:
                    sufficient = runs.sufficient(found, seed=seed)
                    counts.update(sufficient=len(sufficient),
                                  beyond=len(found.asked - sufficient))
                if found.unneeded:
                    wasteful.append((seed, found))
        finally:
            runs.close()
    columns = ["queries", "plain", "needed", "asked", "unneeded", "missed"]
    if options.sufficient:
        columns += ["sufficient", "beyond"]
    print(f"{'shape':22s}" + "".join(f"{name:>11s}" for name in columns))
    for name, counts in sorted(totals.items(), key=lambda item: -item[1]["unneeded"]):
        print(f"{name:22s}" + "".join(f"{counts[column]:11d}" for column in columns))
    whole = sum(totals.values(), Counter())
    print(f"{'all':22s}" + "".join(f"{whole[column]:11d}" for column in columns))
    print()
    for seed, found in wasteful:
        by_call = Counter(f"{kind} {question}" for kind, question, _ in found.unneeded)
        print(f"seed {seed}: {found.sql}\n    unneeded: {dict(by_call)}")


if __name__ == "__main__":
    main()
