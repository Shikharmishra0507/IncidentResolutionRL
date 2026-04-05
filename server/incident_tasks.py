"""Incident resolution tasks with grading criteria."""

from enum import Enum
from typing import Callable

try:
    from ..models import IncidentState, IncidentStatus, FailureCategory, Remediation
except ImportError:
    from models import IncidentState, IncidentStatus, FailureCategory, Remediation


class DifficultyLevel(str, Enum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class IncidentTask:
    """Defines a concrete incident resolution task with grading criteria."""

    def __init__(
        self,
        name: str,
        description: str,
        incident_title: str,
        incident_description: str,
        difficulty: DifficultyLevel,
        budget: int,
        num_services: int,
        root_cause_service: str,
        root_cause_category: FailureCategory,
        correct_remediation: Remediation,
        services: list[dict],
        dependencies: dict[str, list[str]],
        grader: Callable[[IncidentState, list[str]], float],
    ):
        """
        Initialize a task.

        Args:
            name: Task name
            description: Task description
            incident_title: Title of the incident
            incident_description: Full incident description
            difficulty: Difficulty level (easy/medium/hard)
            budget: Maximum steps allowed
            num_services: Number of services in topology
            root_cause_service: Service with the root cause
            root_cause_category: Type of failure
            correct_remediation: Correct remediation action
            services: List of service definitions
            dependencies: Service dependency graph
            grader: Function that takes (state, actions_taken) -> score (0.0-1.0)
        """
        self.name = name
        self.description = description
        self.incident_title = incident_title
        self.incident_description = incident_description
        self.difficulty = difficulty
        self.budget = budget
        self.num_services = num_services
        self.root_cause_service = root_cause_service
        self.root_cause_category = root_cause_category
        self.correct_remediation = correct_remediation
        self.services = services
        self.dependencies = dependencies
        self.grader = grader

    def grade(self, state: IncidentState, actions_taken: list[str]) -> float:
        """Evaluate solution quality."""
        return self.grader(state, actions_taken)


# Helper function for comprehensive grading
def comprehensive_grader(task: IncidentTask, state: IncidentState, actions_taken: list[str]) -> float:
    """Comprehensive grading function that evaluates multiple aspects."""
    score = 0.0

    # Root cause service accuracy (0.20 max)
    if hasattr(state, 'submitted_diagnosis') and state.submitted_diagnosis:
        diagnosis = state.submitted_diagnosis
        if diagnosis.get('service') == task.root_cause_service:
            score += 0.20
        elif diagnosis.get('service') in task.dependencies.get(task.root_cause_service, []):
            score += 0.10  # One hop away

    # Root cause category accuracy (0.15 max)
    if hasattr(state, 'submitted_diagnosis') and state.submitted_diagnosis:
        diagnosis = state.submitted_diagnosis
        if diagnosis.get('category') == task.root_cause_category:
            score += 0.15
        elif diagnosis.get('category') and diagnosis.get('category').split('_')[0] == str(task.root_cause_category).split('_')[0]:
            score += 0.07  # Same failure family

    # Remediation accuracy (0.10 max)
    if hasattr(state, 'submitted_diagnosis') and state.submitted_diagnosis:
        diagnosis = state.submitted_diagnosis
        if diagnosis.get('remediation') == task.correct_remediation:
            score += 0.10
        elif diagnosis.get('remediation') and any(rem in str(diagnosis.get('remediation')).lower() for rem in str(task.correct_remediation).lower().split('_')):
            score += 0.05  # Partial credit

    # Evidence quality (0.10 max) - % of causal chain investigated
    causal_chain = [task.root_cause_service]
    current = task.root_cause_service
    while current in task.dependencies:
        for dep in task.dependencies[current]:
            if dep not in causal_chain:
                causal_chain.append(dep)
                current = dep
                break
        else:
            break

    investigated_services = set()
    for action in actions_taken:
        if action in ['query_logs', 'query_metrics', 'check_deploys', 'check_status', 'inspect_code']:
            # Extract service from action parameters if available
            pass  # Simplified - would need action parameter tracking

    evidence_score = min(0.10, 0.10 * (len(investigated_services) / len(causal_chain)))
    score += evidence_score

    # Efficiency (0.15 max) - fewer steps = higher score
    efficiency_ratio = max(0, 1 - (len(actions_taken) / task.budget))
    score += 0.15 * efficiency_ratio

    # Penalties (-0.35 max)
    penalties = 0.0

    # Shotgun diagnosis penalty
    if len(actions_taken) > task.budget * 0.8 and not hasattr(state, 'submitted_diagnosis'):
        penalties -= 0.15

    # Circular investigation penalty
    action_counts = {}
    for action in actions_taken:
        action_counts[action] = action_counts.get(action, 0) + 1
    if any(count > 3 for count in action_counts.values()):
        penalties -= 0.10

    # Red herring chasing penalty (would need to track which services are red herrings)
    penalties -= min(0.10, 0.02 * len([a for a in actions_taken if 'red_herring' in str(a)]))

    score = max(0.0, min(1.0, score + penalties))

    return score


# Task definitions with service topologies
INCIDENT_TASKS = [
    # EASY TASK: 3 services, 10 steps, single service issue
    IncidentTask(
        name="Bad Deploy Causes NullPointerException",
        description="Bad deploy causes NullPointerException in API service",
        incident_title="API Service NullPointerException",
        incident_description="API service throwing NullPointerException after recent deploy. Users experiencing 500 errors.",
        difficulty=DifficultyLevel.EASY,
        budget=10,
        num_services=3,
        root_cause_service="api-service",
        root_cause_category=FailureCategory.BAD_DEPLOY,
        correct_remediation=Remediation.ROLLBACK_DEPLOY,
        services=[
            {"name": "api-service", "type": "web", "version": "2.1.0"},
            {"name": "db-service", "type": "database", "version": "1.8.2"},
            {"name": "cache-service", "type": "cache", "version": "3.0.1"}
        ],
        dependencies={
            "api-service": ["db-service", "cache-service"],
            "db-service": [],
            "cache-service": []
        },
        grader=lambda state, actions: comprehensive_grader(INCIDENT_TASKS[0], state, actions),
    ),

    # MEDIUM TASK: 5 services, 15 steps, 3-hop causal chain
    IncidentTask(
        name="Config Change Cascades Through Dependencies",
        description="Config change in upstream service cascades through dependency chain",
        incident_title="Database Connection Timeout Cascade",
        incident_description="Database connection timeouts propagating through service mesh. Multiple services affected.",
        difficulty=DifficultyLevel.MEDIUM,
        budget=15,
        num_services=5,
        root_cause_service="config-service",
        root_cause_category=FailureCategory.CONFIG_CHANGE,
        correct_remediation=Remediation.FIX_CONFIG,
        services=[
            {"name": "api-gateway", "type": "gateway", "version": "1.2.0"},
            {"name": "user-service", "type": "web", "version": "2.0.5"},
            {"name": "order-service", "type": "web", "version": "1.9.1"},
            {"name": "db-service", "type": "database", "version": "1.8.2"},
            {"name": "config-service", "type": "config", "version": "1.1.0"}
        ],
        dependencies={
            "api-gateway": ["user-service", "order-service"],
            "user-service": ["db-service", "config-service"],
            "order-service": ["db-service", "config-service"],
            "db-service": ["config-service"],
            "config-service": []
        },
        grader=lambda state, actions: comprehensive_grader(INCIDENT_TASKS[1], state, actions),
    ),

    # HARD TASK: 8 services, 20 steps, 4-hop chain with red herrings
    IncidentTask(
        name="Memory Leak with Multiple Red Herrings",
        description="Memory leak in core service with multiple red herrings and complex dependency chain",
        incident_title="System-wide Performance Degradation",
        incident_description="Gradual performance degradation across all services. Memory usage climbing, response times increasing.",
        difficulty=DifficultyLevel.HARD,
        budget=20,
        num_services=8,
        root_cause_service="core-service",
        root_cause_category=FailureCategory.RESOURCE_EXHAUSTION,
        correct_remediation=Remediation.SCALE_UP,
        services=[
            {"name": "load-balancer", "type": "lb", "version": "2.1.0"},
            {"name": "api-service", "type": "web", "version": "2.1.0"},
            {"name": "auth-service", "type": "auth", "version": "1.5.2"},
            {"name": "core-service", "type": "core", "version": "3.0.0"},
            {"name": "worker-service", "type": "worker", "version": "2.2.1"},
            {"name": "cache-service", "type": "cache", "version": "3.0.1"},
            {"name": "db-service", "type": "database", "version": "1.8.2"},
            {"name": "monitoring-service", "type": "monitoring", "version": "1.3.0"}
        ],
        dependencies={
            "load-balancer": ["api-service"],
            "api-service": ["auth-service", "core-service"],
            "auth-service": ["core-service", "cache-service"],
            "core-service": ["worker-service", "db-service"],
            "worker-service": ["db-service", "cache-service"],
            "cache-service": ["db-service"],
            "db-service": [],
            "monitoring-service": ["core-service", "worker-service"]
        },
        grader=lambda state, actions: comprehensive_grader(INCIDENT_TASKS[2], state, actions),
    ),
]
