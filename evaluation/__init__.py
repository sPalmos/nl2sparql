"""Evaluation support package for NL-to-SPARQL.

The evaluation flow is:

1. `dataset_builder` helps create or validate dataset JSON files by executing
   gold SPARQL against a known endpoint.
2. `dataset_schema` defines the Pydantic models for dataset inputs and
   experiment outputs.
3. `experiment_runner` runs each natural-language question through the runtime
   pipeline for an active ontology package.
4. `answer_comparison` normalizes and compares generated answer rows against
   gold rows.
5. `metrics` aggregates question-level outcomes into run-level reports.

The top-level CLI entry point is `evaluate.py`, which delegates to
`experiment_runner.run_from_cli()`.
"""
