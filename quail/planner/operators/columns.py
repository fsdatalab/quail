"""The planner of each column operator kind, by its logical node class."""

from quail.logical import (
    SemanticClassify,
    SemanticExtract,
    SemanticMap,
    SemanticSaeFeatures,
)
from quail.planner.operators.classify import ClassifyPlanner
from quail.planner.operators.common import ColumnPlanner
from quail.planner.operators.extract import ExtractPlanner
from quail.planner.operators.map import MapPlanner
from quail.planner.operators.sae_features import SaeFeaturesPlanner

COLUMN_PLANNERS = {
    SemanticClassify: ClassifyPlanner(),
    SemanticExtract: ExtractPlanner(),
    SemanticSaeFeatures: SaeFeaturesPlanner(),
    SemanticMap: MapPlanner(),
}


def column_planner(node) -> ColumnPlanner:
    """Return the planner of a logical column operator."""
    return COLUMN_PLANNERS[type(node)]
