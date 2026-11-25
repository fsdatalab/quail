"""Model operators, one class per physical node type."""


def operator_types() -> dict:
    """Return the ModelOperator class of each node type the GPU runs."""
    from quail.backends.quail.executor.operators.classify import ClassifyOperator
    from quail.backends.quail.executor.operators.extract import ExtractOperator
    from quail.backends.quail.executor.operators.filter import (
        ConditionGate,
        FilterOperator,
    )
    from quail.backends.quail.executor.operators.join import JoinOperator
    from quail.backends.quail.executor.operators.map import MapOperator
    from quail.backends.quail.executor.operators.sae_features import (
        SaeFeaturesOperator,
    )
    from quail.backends.quail.executor.operators.score import ScoreOperator
    from quail.physical import (
        AiClassify,
        AiExtract,
        AiFilter,
        AiJoin,
        AiMap,
        AiSaeFeatures,
        AiScore,
        Filter,
    )

    return {AiFilter.type_name: FilterOperator,
            AiClassify.type_name: ClassifyOperator,
            AiExtract.type_name: ExtractOperator, AiJoin.type_name: JoinOperator,
            AiSaeFeatures.type_name: SaeFeaturesOperator,
            AiMap.type_name: MapOperator, AiScore.type_name: ScoreOperator,
            Filter.type_name: ConditionGate}
