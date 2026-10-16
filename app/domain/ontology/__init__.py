"""Public ontology package APIs."""

from app.domain.ontology.package_activation import ActivationResult, activate_package
from app.domain.ontology.onboarding_workflow import OnboardingResult, onboard_ontology_file, onboard_sparql_endpoint

__all__ = [
    "ActivationResult",
    "OnboardingResult",
    "activate_package",
    "onboard_ontology_file",
    "onboard_sparql_endpoint",
]
