from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class TaskMetricSpec:
    primary: str
    secondary: Optional[str]
    higher_is_better: bool = True


# Mapping based on the GLUE benchmark definitions (CoLA=MCC; MRPC/QQP=F1/Acc; STS-B=Pearson/Spearman; others=Accuracy)
GLUE_TASK_METRICS: Dict[str, TaskMetricSpec] = {
    # binary acceptability: matthews_correlation
    "cola": TaskMetricSpec(primary="eval_matthews_correlation", secondary=None, higher_is_better=True),
    # paraphrase: F1 primary, accuracy secondary
    "mrpc": TaskMetricSpec(primary="eval_f1", secondary="eval_accuracy", higher_is_better=True),
    "qqp": TaskMetricSpec(primary="eval_f1", secondary="eval_accuracy", higher_is_better=True),
    # natural language inference variants
    "mnli": TaskMetricSpec(primary="eval_accuracy", secondary=None, higher_is_better=True),
    "qnli": TaskMetricSpec(primary="eval_accuracy", secondary=None, higher_is_better=True),
    "rte": TaskMetricSpec(primary="eval_accuracy", secondary=None, higher_is_better=True),
    "wnli": TaskMetricSpec(primary="eval_accuracy", secondary=None, higher_is_better=True),
    # sentiment
    "sst2": TaskMetricSpec(primary="eval_accuracy", secondary=None, higher_is_better=True),
    # semantic textual similarity: Pearson primary, Spearman secondary
    "stsb": TaskMetricSpec(primary="eval_pearson", secondary="eval_spearman", higher_is_better=True),
}


def normalize_dataset_name(dataset: str) -> str:
    """Extracts the GLUE task name from typical dataset identifiers.
    Examples:
      glue-tvt_rte -> rte
      glue-tvt_cola -> cola
      rte -> rte
    """
    s = dataset.lower()
    if s.startswith("glue"):
        parts = s.split("_")
        if len(parts) >= 2:
            return parts[1]
    return s


def get_task_spec(dataset: str) -> TaskMetricSpec:
    task = normalize_dataset_name(dataset)
    if task not in GLUE_TASK_METRICS:
        raise KeyError(f"Unsupported GLUE task for aggregation: {dataset} (normalized: {task})")
    return GLUE_TASK_METRICS[task]


def select_score(row: Dict[str, Any], spec: TaskMetricSpec) -> Optional[float]:
    """Return the score from a log_history row per task metric spec, or None if absent."""
    if spec.primary in row and row[spec.primary] is not None:
        return float(row[spec.primary])
    if spec.secondary:
        # handle STS-B spearman naming variants
        sec = spec.secondary
        if sec not in row and sec == "eval_spearman" and ("eval_spearmanr" in row):
            sec = "eval_spearmanr"
        if (sec in row) and row[sec] is not None:
            return float(row[sec])
    return None


