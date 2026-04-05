"""Incident Response Environment Implementation."""

from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.client_types import StepResult

from rl_demo.models import FailureCategory

try:
    from ..models import IncidentAction, IncidentObservation, IncidentState, IncidentStatus
except ImportError:
    from models import IncidentAction, IncidentObservation, IncidentState, IncidentStatus

from .incident_tasks import INCIDENT_TASKS
from .reward_function import RewardFunction



class IncidentEnvironment(Environment):
    """
    Incident Response Learning Environment.

    This environment simulates an incident response scenario where an agent must:
    - Analyze incident description
    - Fetch logs, check metrics, search knowledge base
    - Take resolution actions (restart, scale, resolve, escalate)
    - Complete assigned task within step budget
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the incident environment."""
        self._state = None
        self._reset_count = 0
        self._current_task = None
        self._reward_function = RewardFunction()

    def reset(self) -> IncidentObservation:
        """
        Reset the environment with a random task.

        Returns:
            IncidentObservation with initial incident state
        """
        self._reset_count += 1

        # Select random task
        self._current_task = random.choice(INCIDENT_TASKS)

        self._state = IncidentState(
            episode_id=int(uuid4().int % 1000000),
            incident_id=f"INC-{self._reset_count:05d}",
            title=self._current_task.incident_title,
            description=self._current_task.incident_description,
            severity="high",
            status=IncidentStatus.OPEN,
            services=self._current_task.services,
            logs=None,
            metrics=None,
            deploys=None,
            dependencies=self._current_task.dependencies,
            status_info=None,
            code_inspection=None,
            kb_results=None,
            actions_taken=[],
            step_count=0,
            budget=self._current_task.budget,
            root_cause_service=self._current_task.root_cause_service,
            root_cause_category=self._current_task.root_cause_category,
            correct_remediation=self._current_task.correct_remediation,
        )

        return IncidentObservation(
            incident_id=self._state.incident_id,
            title=self._state.title,
            description=self._state.description,
            severity=self._state.severity,
            status=self._state.status,
            services=self._state.services,
            logs=None,
            metrics=None,
            deploys=None,
            dependencies=self._state.dependencies,
            status_info=None,
            code_inspection=None,
            kb_results=None,
            actions_taken=[],
            step_count=0,
            budget=self._state.budget,
        )

    def step(self, action: IncidentAction) -> StepResult[IncidentObservation]:
        """
        Execute a step in the environment.

        Args:
            action: IncidentAction with action_type and parameters

        Returns:
            StepResult with observation, reward, done, and metadata

        Raises:
            RuntimeError: If environment not initialized (reset not called)
            ValueError: If action type or parameters are invalid
        """
        # Validate environment state
        if self._state is None or self._current_task is None:
            raise RuntimeError(
                "Environment not initialized. Call reset() before step()."
            )

        # Validate action type
        valid_actions = {
            "query_logs", "query_metrics", "check_deploys", "trace_dependencies",
            "check_status", "inspect_code", "search_kb", "submit_diagnosis"
        }
        if action.action_type not in valid_actions:
            raise ValueError(f"Invalid action type: {action.action_type}")

        self._state.step_count += 1
        self._state.actions_taken.append(action.action_type)

        # Simulate action outcomes based on action type
        if action.action_type == "query_logs":
            service_name = action.parameters.get("service_name", "api-service")
            self._state.logs = self._generate_logs_for_service(service_name)

        elif action.action_type == "query_metrics":
            service_name = action.parameters.get("service_name", "api-service")
            self._state.metrics = self._generate_metrics_for_service(service_name)

        elif action.action_type == "check_deploys":
            service_name = action.parameters.get("service_name", "api-service")
            self._state.deploys = self._generate_deploy_history(service_name)

        elif action.action_type == "trace_dependencies":
            service_name = action.parameters.get("service_name", "api-service")
            # Dependencies are already available in state
            pass

        elif action.action_type == "check_status":
            self._state.status_info = self._generate_status_info()

        elif action.action_type == "inspect_code":
            service_name = action.parameters.get("service_name", "api-service")
            self._state.code_inspection = self._generate_code_inspection(service_name)

        elif action.action_type == "search_kb":
            query = action.parameters.get("query", "")
            self._state.kb_results = self._generate_kb_search_results(query)

        elif action.action_type == "submit_diagnosis":
            # Handle diagnosis submission
            diagnosis = {
                "service": action.parameters.get("service"),
                "category": action.parameters.get("category"),
                "remediation": action.parameters.get("remediation"),
                "step_count": self._state.step_count
            }
            self._state.submitted_diagnosis = diagnosis
            # Diagnosis submitted - episode can end

        # Compute reward and done status
        reward, done = self._reward_function.compute_reward(
            action.action_type, self._state, self._current_task
        )   

        observation = IncidentObservation(
            incident_id=self._state.incident_id,
            title=self._state.title,
            description=self._state.description,
            severity=self._state.severity,
            status=self._state.status,
            services=self._state.services,
            logs=self._state.logs,
            metrics=self._state.metrics,
            deploys=self._state.deploys,
            dependencies=self._state.dependencies,
            status_info=self._state.status_info,
            code_inspection=self._state.code_inspection,
            kb_results=self._state.kb_results,
            actions_taken=self._state.actions_taken,
            step_count=self._state.step_count,
            budget=self._state.budget,
        )
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            metadata={},
        )

    @property
    def state(self) -> IncidentState:
        """Get the current environment state."""
        return self._state

    def _generate_logs_for_service(self, service_name: str) -> list[str]:
        """Generate realistic logs for a service."""
        base_logs = [
            f"INFO: {service_name} started successfully",
            f"INFO: {service_name} connected to dependencies",
        ]

        # Add error logs based on the root cause
        if service_name == self._current_task.root_cause_service:
            if self._current_task.root_cause_category == FailureCategory.BAD_DEPLOY:
                base_logs.extend([
                    f"ERROR: {service_name} NullPointerException in version {self._get_service_version(service_name)}",
                    f"ERROR: {service_name} Failed to initialize component",
                ])
            elif self._current_task.root_cause_category == FailureCategory.RESOURCE_EXHAUSTION:
                base_logs.extend([
                    f"WARNING: {service_name} Memory usage at 95%",
                    f"ERROR: {service_name} OutOfMemoryError",
                ])
            elif self._current_task.root_cause_category == FailureCategory.CONFIG_CHANGE:
                base_logs.extend([
                    f"ERROR: {service_name} Configuration validation failed",
                    f"ERROR: {service_name} Unable to connect to service",
                ])

        return base_logs

    def _generate_metrics_for_service(self, service_name: str) -> dict[str, float]:
        """Generate metrics for a service."""
        base_metrics = {
            "cpu_usage": 45.0 + random.uniform(-10, 20),
            "memory_usage": 60.0 + random.uniform(-15, 25),
            "latency_p95": 100.0 + random.uniform(-20, 100),
            "error_rate": 0.01 + random.uniform(0, 0.1),
            "request_rate": 100.0 + random.uniform(-50, 200),
        }

        # Adjust metrics based on root cause
        if service_name == self._current_task.root_cause_service:
            if self._current_task.root_cause_category == FailureCategory.RESOURCE_EXHAUSTION:
                base_metrics["memory_usage"] = 95.0 + random.uniform(0, 5)
                base_metrics["cpu_usage"] = 85.0 + random.uniform(0, 10)
            elif self._current_task.root_cause_category == FailureCategory.CONFIG_CHANGE:
                base_metrics["error_rate"] = 0.5 + random.uniform(0, 0.3)
                base_metrics["latency_p95"] = 5000.0 + random.uniform(0, 2000)

        return base_metrics

    def _generate_deploy_history(self, service_name: str) -> list[dict]:
        """Generate deployment history for a service."""
        version = self._get_service_version(service_name)
        deploys = [
            {
                "version": version,
                "timestamp": "2024-01-15T10:30:00Z",
                "status": "success",
                "deployer": "jenkins-ci"
            }
        ]

        # Add recent failed deploy if this is the root cause service
        if service_name == self._current_task.root_cause_service and self._current_task.root_cause_category == FailureCategory.BAD_DEPLOY:
            deploys.insert(0, {
                "version": f"{version}-bad",
                "timestamp": "2024-01-15T14:20:00Z",
                "status": "success",  # Deploy succeeded but code is bad
                "deployer": "jenkins-ci"
            })

        return deploys

    def _generate_status_info(self) -> dict[str, str]:
        """Generate status information for all services."""
        status_info = {}
        for service in self._current_task.services:
            service_name = service["name"]
            # Most services are healthy
            status = "healthy"

            # Root cause service might be degraded
            if service_name == self._current_task.root_cause_service:
                if self._current_task.root_cause_category in [FailureCategory.RESOURCE_EXHAUSTION, FailureCategory.DEPENDENCY_FAILURE]:
                    status = "degraded"
                elif self._current_task.root_cause_category == FailureCategory.CONFIG_CHANGE:
                    status = "error"

            status_info[service_name] = status

        return status_info

    def _generate_code_inspection(self, service_name: str) -> dict:
        """Generate code inspection results for a service."""
        inspection = {
            "language": "java" if service_name.endswith("-service") else "python",
            "lines_of_code": random.randint(1000, 50000),
            "complexity_score": random.uniform(1.0, 10.0),
            "test_coverage": random.uniform(0.3, 0.95),
            "vulnerabilities": random.randint(0, 5),
        }

        # Add specific issues for root cause service
        if service_name == self._current_task.root_cause_service:
            if self._current_task.root_cause_category == FailureCategory.BAD_DEPLOY:
                inspection["recent_changes"] = [
                    "Modified null checking logic",
                    "Updated dependency injection",
                    "Changed error handling"
                ]
                inspection["potential_issues"] = ["Null pointer dereference risk"]

        return inspection

    def _generate_kb_search_results(self, query: str) -> list[str]:
        """Generate knowledge base search results."""
        # Base KB articles
        kb_articles = [
            "Memory leaks typically show gradual performance degradation over time",
            "Database connection pool exhaustion causes timeout errors",
            "Bad deploys can introduce null pointer exceptions",
            "Config changes require service restarts to take effect",
            "High latency often indicates resource constraints or dependency issues",
        ]

        # Add relevant articles based on root cause
        if self._current_task.root_cause_category == FailureCategory.BAD_DEPLOY:
            kb_articles.extend([
                "Recent code deployments should be rolled back if errors appear immediately",
                "NullPointerException often indicates incomplete null checks in new code",
            ])
        elif self._current_task.root_cause_category == FailureCategory.RESOURCE_EXHAUSTION:
            kb_articles.extend([
                "Memory exhaustion requires horizontal scaling or memory optimization",
                "Gradual performance degradation suggests resource leaks",
            ])
        elif self._current_task.root_cause_category == FailureCategory.CONFIG_CHANGE:
            kb_articles.extend([
                "Configuration changes cascade through dependent services",
                "Service restarts required after config updates",
            ])

        # Return top 3 most relevant articles
        return kb_articles[:3]

    def _get_service_version(self, service_name: str) -> str:
        """Get version for a service."""
        for service in self._current_task.services:
            if service["name"] == service_name:
                return service.get("version", "1.0.0")
        return "1.0.0"


