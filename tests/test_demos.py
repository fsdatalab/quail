"""The CPU walkthrough demo keeps running."""

import io
from contextlib import redirect_stdout

from demos import plan_walkthrough


def test_plan_walkthrough_prints_the_same_row_from_every_plan():
    out = io.StringIO()
    with redirect_stdout(out):
        plan_walkthrough.main()
    text = out.getvalue()
    assert text.count("[{'c.id': 'c0', 'e.id': 'e0'}]") == 3
    assert "'ai_filter:c', 'barrier:c', 'ai_join:c'" in text
    assert "PlanEditError" in text
