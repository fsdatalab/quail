"""Operator implementations for the shared document scheduler."""


def execution_parts() -> dict:
    """Return the execution implementation for each supported node type."""
    from quail.backends.quail.executor.operators.apply import ApplyGate
    from quail.backends.quail.executor.operators.classify import ClassifyPart
    from quail.backends.quail.executor.operators.extract import ExtractPart
    from quail.backends.quail.executor.operators.filter import ConditionGate, FilterPart
    from quail.backends.quail.executor.operators.join import JoinPart
    from quail.backends.quail.executor.operators.sae_features import (
        SaeFeaturesPart,
    )
    from quail.backends.quail.executor.operators.score import ScoreBatch
    from quail.physical import (
        AiClassify,
        AiExtract,
        AiFilter,
        AiJoin,
        AiSaeFeatures,
        AiScore,
        Filter,
        Foreign,
    )

    return {AiFilter.type_name: FilterPart, AiClassify.type_name: ClassifyPart,
            AiExtract.type_name: ExtractPart, AiJoin.type_name: JoinPart,
            AiSaeFeatures.type_name: SaeFeaturesPart,
            AiScore.type_name: ScoreBatch,
            Filter.type_name: ConditionGate, Foreign.type_name: ApplyGate}
