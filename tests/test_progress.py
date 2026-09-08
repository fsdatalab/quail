"""Progress lines and the quiet() switch."""

from quail.progress import Progress, quiet, say


def test_quiet_suppresses_progress_lines(capsys):
    say("shown")
    with quiet():
        say("hidden")
        Progress("hidden step", total=2).finish("hidden step done")
    Progress("step", total=2).finish("step done", "extra")
    out = capsys.readouterr().out
    assert "quail: shown" in out
    assert "hidden" not in out
    assert "quail: step done: 0/2 documents" in out
    assert out.strip().endswith("documents/s, extra")
