"""Measure SGLang request preparation before each join reaches the GPU."""

import cProfile
import inspect
import pstats
import time
from collections import defaultdict
from functools import wraps

PREDICTION_TEXT = (
    "Preparing and sending all tokenized join requests will account for most "
    "of SGLang's 12 to 14 seconds before GPU work begins. Profile Python "
    "calls during preparation for join 1; use lightweight timers for joins "
    "2 and 3. Answers and token counts should match the saved baseline."
)


class InputProfiler:
    """Record elapsed request preparation time and selected Python call costs."""

    def __init__(self):
        from sglang.srt.managers.io_struct import GenerateReqInput
        from sglang.srt.managers.tokenizer_manager import TokenizerManager

        self.number = -1
        self.timings = defaultdict(lambda: {"calls": 0, "wall_s": 0.0})
        self.intervals = []
        self.cpu_profile = None
        self.destination = None
        for owner, name, coarse in (
            (GenerateReqInput, "normalize_batch_and_arguments", True),
            (GenerateReqInput, "__getitem__", False),
            (TokenizerManager, "_batch_tokenize_and_process", True),
            (TokenizerManager, "_send_batch_request", True),
            (TokenizerManager, "_dispatch_to_scheduler", True),
            (TokenizerManager, "_create_tokenized_object", False),
            (TokenizerManager, "_validate_one_request", False),
        ):
            self.annotate(owner, name, coarse)

    def reset(self, number, destination):
        """Begin collecting timings for one join."""
        self.number = number
        self.destination = destination
        self.timings.clear()
        self.intervals.clear()
        self.cpu_profile = None

    def annotate(self, owner, name, coarse):
        """Wrap one preparation method with an elapsed-time measurement."""
        original = getattr(owner, name)
        label = f"sglang.input.{name}"

        def start():
            import torch

            if self.number < 0:
                return None
            context = torch.profiler.record_function(label) if coarse else None
            if context is not None:
                context.__enter__()
            cpu_profile = None
            if name == "_batch_tokenize_and_process" and self.number == 0:
                cpu_profile = self.cpu_profile = cProfile.Profile()
                cpu_profile.enable()
            return time.perf_counter(), time.time_ns(), context, cpu_profile

        def finish(state):
            if state is None:
                return
            started, started_ns, context, cpu_profile = state
            elapsed = time.perf_counter() - started
            if cpu_profile is not None:
                cpu_profile.disable()
            value = self.timings[label]
            value["calls"] += 1
            value["wall_s"] += elapsed
            if coarse:
                self.intervals.append({"name": label, "start_unix_ns": started_ns,
                                       "end_unix_ns": time.time_ns()})
            if context is not None:
                context.__exit__(None, None, None)

        if inspect.iscoroutinefunction(original):
            @wraps(original)
            async def wrapped(*args, **kwargs):
                state = start()
                try:
                    return await original(*args, **kwargs)
                finally:
                    finish(state)
        else:
            @wraps(original)
            def wrapped(*args, **kwargs):
                state = start()
                try:
                    return original(*args, **kwargs)
                finally:
                    finish(state)
        setattr(owner, name, wrapped)

    def summary(self):
        """Return timing totals and Python function costs for the join."""
        result = {"timings": dict(self.timings), "intervals": list(self.intervals)}
        if self.cpu_profile is not None:
            path = self.destination / "input-preparation.pstats"
            self.cpu_profile.dump_stats(str(path))
            stats = pstats.Stats(self.cpu_profile)
            rows = [
                {"file": key[0], "line": key[1], "function": key[2],
                 "primitive_calls": value[0], "calls": value[1],
                 "self_s": value[2], "cumulative_s": value[3]}
                for key, value in stats.stats.items()
            ]
            result["python_profile_path"] = str(path)
            result["top_self"] = sorted(
                rows, key=lambda r: r["self_s"], reverse=True)[:25]
            result["top_cumulative"] = sorted(
                rows, key=lambda r: r["cumulative_s"], reverse=True)[:25]
        return result


def profile_worker(directory, connection):
    """Run SGLang with GPU traces and detailed input preparation measurements."""
    import experiments.sglang_join_profile_worker as worker

    worker.PREDICTION_TEXT = PREDICTION_TEXT
    worker.profile_worker(directory, connection, input_detail=True)
