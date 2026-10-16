"""Typed data contracts for NL-to-SPARQL evaluation.

This module defines the JSON shape for evaluation datasets and for the result
artifacts written after an experiment run. The rest of the evaluation package
uses these models to keep file I/O, scoring, and reporting aligned.

The input side is `EvaluationDataset`, which contains many
`EvaluationQuestion` records. The output side is `ExperimentRun`, which contains
one `QuestionResult` per question plus the runtime configuration used for the
run.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class QueryShape(str, Enum):
    """Coarse structural shape of the gold SPARQL query.

    These labels are used for grouped metrics, for example execution accuracy
    by `single-edge` versus `chain` questions.
    """

    SINGLE_EDGE = "single-edge"
    CHAIN = "chain"
    STAR = "star"
    TREE = "tree"
    CYCLE = "cycle"
    FLOWER = "flower"


class ComplexityTier(str, Enum):
    """Human-assigned difficulty tier for a question."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class QuestionType(str, Enum):
    """Expected answer format for the question."""

    LIST = "list"
    BOOLEAN = "boolean"
    COUNT = "count"
    FACTOID = "factoid"


class DatasetSource(str, Enum):
    """Origin or benchmark family for the dataset."""

    CUSTOM = "custom"
    QALD = "qald"
    SPIDER4SPARQL = "spider4sparql"
    LCQUAD = "lcquad"


class EvaluationQuestion(BaseModel):
    """One benchmark item.

    `nl_question` is sent to the runtime pipeline. `gold_sparql` is the
    reference query used to build or validate `gold_answers`. Non-empty
    `gold_answers` are the expected rows. An empty list is a scored expected
    empty result set.
    """

    id: str = Field(..., description="Unique question identifier, e.g. Q001")
    nl_question: str = Field(..., description="The natural-language question")
    gold_sparql: str = Field(..., description="Gold-standard SPARQL query")
    gold_answers: list[dict[str, str]] = Field(
        default_factory=list,
        description="Expected result rows. Empty means the correct result set has no rows.",
    )
    complexity_tier: ComplexityTier = Field(..., description="Broad complexity category")
    query_shape: QueryShape = Field(..., description="Structural query shape")
    question_type: QuestionType = Field(..., description="Expected answer format")
    notes: Optional[str] = Field(default=None, description="Optional question notes")


class EvaluationDataset(BaseModel):
    """A complete set of evaluation questions for one ontology or endpoint."""

    dataset_name: str = Field(..., description="Unique dataset identifier")
    ontology_file: str = Field(..., description="Ontology filename used by the dataset")
    source: DatasetSource = Field(..., description="Dataset origin")
    description: Optional[str] = Field(default=None, description="Dataset description")
    questions: list[EvaluationQuestion] = Field(..., min_length=1)

    @property
    def size(self) -> int:
        return len(self.questions)

    def by_complexity(self, tier: ComplexityTier) -> list[EvaluationQuestion]:
        return [q for q in self.questions if q.complexity_tier == tier]

    def by_shape(self, shape: QueryShape) -> list[EvaluationQuestion]:
        return [q for q in self.questions if q.query_shape == shape]

    def by_type(self, qtype: QuestionType) -> list[EvaluationQuestion]:
        return [q for q in self.questions if q.question_type == qtype]


class IterationLog(BaseModel):
    """Compact log of one runtime attempt for one evaluated question.

    The runtime trace is richer; this model keeps only the fields needed by
    metrics and readable evaluation reports.
    """

    iteration: int
    generated_sparql: str
    validation_stages: dict[str, bool] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    execution_status: str | None = None


class QuestionResult(BaseModel):
    """Evaluation output for one question.

    This stores the final SPARQL candidate, generated answer rows, scoring
    status, comparison details, attempt logs, latency, pipeline configuration,
    and links to the runtime traces that explain how the answer was produced.
    """

    question_id: str
    nl_question: str
    gold_sparql: str
    gold_answers: list[dict[str, str]] = Field(default_factory=list)

    final_sparql: Optional[str] = None
    final_answers: Optional[list[dict[str, str]]] = None
    status: str = "pending"
    errors: list[str] = Field(default_factory=list)

    scoring_status: str = "scored"
    is_scored: bool = True
    comparison: dict[str, object] | None = None

    iterations: list[IterationLog] = Field(default_factory=list)
    total_iterations: int = 0

    total_latency_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    model_name: str = ""
    pipeline_config: dict[str, object] = Field(default_factory=dict)
    trace_path: str | None = None
    readable_trace_path: str | None = None


class ExperimentRun(BaseModel):
    """All question results for one dataset run against one ontology package."""

    experiment_id: str
    dataset_name: str
    package_dir: str
    model_name: str
    pipeline_config: dict[str, object] = Field(default_factory=dict)
    results: list[QuestionResult] = Field(default_factory=list)
    timestamp: str = ""
