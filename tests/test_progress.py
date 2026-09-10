"""Progress lines and the quiet() switch."""

from quail.progress import Progress, logger, quiet, say, set_gpu_index


def test_progress_labels_and_quiet_mode(capsys):
    say("shown")
    with quiet():
        say("hidden")
        Progress("hidden step", total=2).finish("hidden step done")
    Progress("step", total=2).finish("step done", "extra")
    out = capsys.readouterr().out
    assert "[quail] shown" in out
    assert "INFO" in out
    assert "hidden" not in out
    assert "[quail] step done: 0/2 documents" in out
    assert out.strip().endswith("documents/s, extra")

    try:
        set_gpu_index(3)
        with quiet():
            progress = Progress("GEMM warmup", total=2, unit="configurations",
                                every=0, emit=logger.info)
            progress.update(2)
            progress.finish("GEMM warmup done")
        Progress("filter (2 stages)", total=10).finish("filter done")
    finally:
        set_gpu_index(None)
    say("parent process")
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4
    assert all("[quail][GPU 3]" in line for line in lines[:3])
    assert "2/2 configurations" in lines[0]
    assert "[quail] parent process" in lines[-1]
