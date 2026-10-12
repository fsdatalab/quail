"""Plan AI.PARSE: refuse what the session's parser cannot run, and split memory.

An OCR parser's weights and arena sit on the GPU beside the reading
model's. The planner gives the reader a device with that memory taken
out, so every budget it computes for the reader leaves room for the
parser.
"""

from dataclasses import dataclass, replace

from quail.cost.budgets import POOL_FRACTION
from quail.planner.plan import Refusal
from quail.specs import DeviceSpec, ModelSpec


@dataclass(frozen=True)
class ParsePlacement:
    """Where the session's parser runs and the memory it takes.

    Attributes:
        parser: The EngineConfig.parser value.
        lanes: Pages an OCR parser decodes at once; 0 for pdfium.
        parser_bytes: GPU memory the parser takes beside the reader.
        reader_device: The device the reader is planned on, with the
            parser's memory taken out.
    """

    parser: str
    lanes: int
    parser_bytes: float
    reader_device: DeviceSpec


def _refusal(reason: str, constraint: str) -> Refusal:
    return Refusal(reasons=(reason,), constraint=constraint, needed=1,
                   available=0, unit="queries")


def place_parser(plan, *, model: ModelSpec, device: DeviceSpec, gpus: int,
                 backend: str, parser: str | None,
                 models) -> ParsePlacement | Refusal | None:
    """Check a plan's parses against the session's parser and place it.

    Args:
        plan: The LogicalPlan.
        model: The reading model.
        device: The device.
        gpus: Model copies.
        backend: The model backend.
        parser: EngineConfig.parser.
        models: The registered ModelSpecs by name.

    Returns:
        None when the plan parses nothing, a Refusal, or the placement.
    """
    from quail.parsers import ocr_memory, parser_spec

    parses = plan.operators().parses
    if not parses:
        return None
    if parser is None:
        return _refusal(
            "AI.PARSE needs a parser; set EngineConfig(parser='pdfium') or "
            "parser='lightonocr-3-0.8b'", "parse_without_parser")
    spec = parser_spec(parser)
    for node in parses:
        if node.call.mode not in spec.modes:
            return _refusal(
                f"AI.PARSE mode {node.call.mode!r} is not supported by parser "
                f"{parser!r}; it supports {list(spec.modes)}",
                "unsupported_parse_mode")
    if gpus > 1:
        return _refusal("AI.PARSE runs on one GPU; set gpus=1",
                        "parse_needs_one_gpu")
    if backend != "quail":
        return _refusal(f"AI.PARSE runs on the quail backend, not {backend!r}",
                        "parse_needs_quail_backend")
    if not spec.gpu:
        return ParsePlacement(parser, 0, 0.0, device)
    free = device.mem_bytes * POOL_FRACTION - model.W_mem
    try:
        lanes, taken = ocr_memory(models[parser], free)
    except ValueError as error:
        return _refusal(str(error), "parser_memory")
    reader = replace(device, mem_bytes=device.mem_bytes - taken / POOL_FRACTION)
    return ParsePlacement(parser, lanes, taken, reader)
