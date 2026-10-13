"""Operator implementations for the shared document scheduler."""


def execution_parts() -> dict:
    """Return the execution implementation for each supported node type."""
    from quail.backends.quail.executor.operators.apply import ApplyGate
    from quail.backends.quail.executor.operators.classify import ClassifyPart
    from quail.backends.quail.executor.operators.extract import ExtractPart
    from quail.backends.quail.executor.operators.filter import FilterPart, LabelGate
    from quail.backends.quail.executor.operators.join import JoinPart
    from quail.physical import AiClassify, AiExtract, AiFilter, AiJoin, Filter, Foreign

    return {AiFilter.type_name: FilterPart, AiClassify.type_name: ClassifyPart,
            AiExtract.type_name: ExtractPart, AiJoin.type_name: JoinPart,
            Filter.type_name: LabelGate, Foreign.type_name: ApplyGate}
