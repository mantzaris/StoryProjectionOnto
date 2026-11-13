"""Experimental conditions and cross-condition integrity controls.

The package keeps construction timing explicit: C0/C1 seal ontologies before a
query, C2 records an empty pre-query inventory and constructs after reveal, and
``A-FixedSelect`` inherits one complete same-seed C1 ontology under a mechanically
selection-only capability set.
"""

from .base import (
    ComparisonInputManifest,
    ConditionAttemptRecord,
    ConditionExecutionTrace,
    ConditionInterface,
    ConditionPreparation,
    ExecutionStage,
    FixedSelectionPreparation,
    ProduceInputs,
    RunConditionConfig,
    SealedPreontology,
    assert_comparison_fairness,
)

__all__ = [
    "ComparisonInputManifest",
    "ConditionAttemptRecord",
    "ConditionExecutionTrace",
    "ConditionInterface",
    "ConditionPreparation",
    "ExecutionStage",
    "FixedSelectionPreparation",
    "ProduceInputs",
    "RunConditionConfig",
    "SealedPreontology",
    "assert_comparison_fairness",
]
