from enum import Enum
from pydantic import BaseModel
from typing import Any, Literal


class IncidentStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class FailureCategory(str, Enum):
    BAD_DEPLOY = "bad_deploy"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIG_CHANGE = "config_change"
    TRAFFIC_SPIKE = "traffic_spike"
    DATA_CORRUPTION = "data_corruption"


class Remediation(str, Enum):
    ROLLBACK_DEPLOY = "rollback_deploy"
    SCALE_UP = "scale_up"
    RESTART_SERVICE = "restart_service"
    FIX_CONFIG = "fix_config"
    ENABLE_RATE_LIMITING = "enable_rate_limiting"
    FAILOVER_TO_BACKUP = "failover_to_backup"


class IncidentAction(BaseModel):
    action_type: Literal[
        "query_logs",
        "query_metrics",
        "check_deploys",
        "trace_dependencies",
        "check_status",
        "inspect_code",
        "search_kb",
        "submit_diagnosis"
    ]
    parameters: dict[str, Any] = {}
    
class IncidentObservation(BaseModel):
    incident_id: str
    title: str
    description: str
    severity: str
    status: IncidentStatus
    services: list[dict[str, Any]] | None = None  # Service topology
    logs: list[str] | None = None
    metrics: dict[str, float] | None = None
    deploys: list[dict[str, Any]] | None = None  # Recent deployments
    dependencies: dict[str, list[str]] | None = None  # Service dependencies
    status_info: dict[str, str] | None = None  # Service health status
    code_inspection: dict[str, Any] | None = None  # Code inspection results
    kb_results: list[str] | None = None  # Knowledge base search results
    actions_taken: list[str]
    step_count: int
    budget: int  # Step budget for this task
    
class IncidentState(BaseModel):
    episode_id: int
    incident_id: str
    title: str
    description: str
    severity: str
    status: IncidentStatus
    services: list[dict[str, Any]] | None = None
    logs: list[str] | None = None
    metrics: dict[str, float] | None = None
    deploys: list[dict[str, Any]] | None = None
    dependencies: dict[str, list[str]] | None = None
    status_info: dict[str, str] | None = None
    code_inspection: dict[str, Any] | None = None
    kb_results: list[str] | None = None
    actions_taken: list[str]
    step_count: int
    budget: int
    # Ground truth for grading
    root_cause_service: str | None = None
    root_cause_category: FailureCategory | None = None
    correct_remediation: Remediation | None = None
    submitted_diagnosis: dict[str, Any] | None = None
