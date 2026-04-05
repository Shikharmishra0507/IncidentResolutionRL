from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import IncidentAction, IncidentObservation, IncidentState


class IncidentEnv(EnvClient[IncidentAction, IncidentObservation, IncidentState]):
    """Client wrapper for the incident response environment."""

    def _step_payload(self, action: IncidentAction) -> Dict:
        """Convert Action to JSON payload for step request."""
        return {
            "action_type": action.action_type,
            "parameters": action.parameters,
        }

    def _parse_result(self, payload: Dict) -> StepResult[IncidentObservation]:
        """Parse the /step response payload into StepResult."""
        obs_data = payload.get("observation", {})

        observation = IncidentObservation(
            incident_id=obs_data.get("incident_id", ""),
            title=obs_data.get("title", ""),
            description=obs_data.get("description", ""),
            severity=obs_data.get("severity", ""),
            status=obs_data.get("status", "open"),
            services=obs_data.get("services"),
            logs=obs_data.get("logs"),
            metrics=obs_data.get("metrics"),
            deploys=obs_data.get("deploys"),
            dependencies=obs_data.get("dependencies"),
            status_info=obs_data.get("status_info"),
            code_inspection=obs_data.get("code_inspection"),
            kb_results=obs_data.get("kb_results"),
            actions_taken=obs_data.get("actions_taken", []),
            step_count=obs_data.get("step_count", 0),
            budget=obs_data.get("budget", 10),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=payload.get("metadata", {}),
        )

    def _parse_state(self, payload: Dict) -> IncidentState:
        """Parse the /state response payload into IncidentState."""
        return IncidentState(
            episode_id=payload.get("episode_id"),
            incident_id=payload.get("incident_id", ""),
            title=payload.get("title", ""),
            description=payload.get("description", ""),
            severity=payload.get("severity", ""),
            status=payload.get("status", "open"),
            services=payload.get("services"),
            logs=payload.get("logs"),
            metrics=payload.get("metrics"),
            deploys=payload.get("deploys"),
            dependencies=payload.get("dependencies"),
            status_info=payload.get("status_info"),
            code_inspection=payload.get("code_inspection"),
            kb_results=payload.get("kb_results"),
            actions_taken=payload.get("actions_taken", []),
            step_count=payload.get("step_count", 0),
            budget=payload.get("budget", 10),
        )
