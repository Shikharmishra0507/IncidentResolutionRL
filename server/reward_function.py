"""Dense reward function for incident response environment."""

try:
    from ..models import IncidentState, IncidentStatus, FailureCategory, Remediation
except ImportError:
    from models import IncidentState, IncidentStatus, FailureCategory, Remediation
from .incident_tasks import IncidentTask

class RewardFunction:
    """Computes dense rewards for agent actions with per-step learning signals."""

    def __init__(self):
        """Initialize reward function with dense reward weights."""
        # Information gain rewards (0.04 to 0.12)
        self.logs_reward = 0.08
        self.metrics_reward = 0.06
        self.deploys_reward = 0.10
        self.status_reward = 0.04
        self.code_inspection_reward = 0.12
        self.dependencies_reward = 0.05

        # Strategy bonuses (0.02 to 0.05)
        self.following_dependencies_bonus = 0.03
        self.cross_referencing_bonus = 0.04
        self.ruling_out_red_herrings_bonus = 0.05

        # Penalties (-0.03 to -0.08)
        self.redundant_queries_penalty = -0.03
        self.chasing_red_herrings_penalty = -0.05
        self.circular_investigation_penalty = -0.08

        # Terminal rewards
        self.correct_diagnosis_reward = 1.0
        self.incorrect_diagnosis_penalty = -0.5

    def compute_reward(
        self,
        action_type: str,
        state: IncidentState,
        task: IncidentTask,
    ) -> tuple[float, bool]:
        """
        Compute dense reward and done flag for current action.

        Args:
            action_type: Type of action taken
            state: Current environment state
            task: Task being solved

        Returns:
            (reward, done) - dense reward signal and episode termination flag
        """
        reward = 0.0
        done = False

        # Information gathering rewards
        if action_type == "query_logs":
            reward += self._reward_information_gain("logs", state, task)
        elif action_type == "query_metrics":
            reward += self._reward_information_gain("metrics", state, task)
        elif action_type == "check_deploys":
            reward += self._reward_information_gain("deploys", state, task)
        elif action_type == "check_status":
            reward += self._reward_information_gain("status", state, task)
        elif action_type == "inspect_code":
            reward += self._reward_information_gain("code", state, task)
        elif action_type == "search_kb":
            reward += self._reward_information_gain("kb", state, task)
        elif action_type == "trace_dependencies":
            reward += self._reward_information_gain("dependencies", state, task)

        # Strategy bonuses
        reward += self._reward_strategy_bonuses(action_type, state, task)

        # Penalties
        reward += self._reward_penalties(action_type, state, task)

        # Diagnosis submission handling
        if action_type == "submit_diagnosis":
            diagnosis_reward, diagnosis_done = self._evaluate_diagnosis(state, task)
            reward += diagnosis_reward
            done = diagnosis_done

        # Episode termination conditions
        if state.step_count >= task.budget:
            done = True
            if not hasattr(state, 'submitted_diagnosis') or not state.submitted_diagnosis:
                reward += self.incorrect_diagnosis_penalty

        return reward, done

    def _reward_information_gain(self, info_type: str, state: IncidentState, task: IncidentTask) -> float:
        """Reward information gathering based on relevance and novelty."""
        base_reward = {
            "logs": self.logs_reward,
            "metrics": self.metrics_reward,
            "deploys": self.deploys_reward,
            "status": self.status_reward,
            "code": self.code_inspection_reward,
            "kb": 0.09,  # Knowledge base search reward
            "dependencies": self.dependencies_reward,
        }.get(info_type, 0.0)

        # Boost reward if this information is relevant to the root cause
        service_param = getattr(state, 'last_action_params', {}).get('service_name')
        if service_param == task.root_cause_service:
            base_reward *= 1.5  # Higher reward for investigating the right service

        # Reduce reward for redundant queries
        if self._is_redundant_query(info_type, state):
            base_reward *= 0.5

        return base_reward

    def _reward_strategy_bonuses(self, action_type: str, state: IncidentState, task: IncidentTask) -> float:
        """Reward strategic investigation patterns."""
        bonus = 0.0

        # Following dependency chains
        if self._is_following_dependencies(action_type, state, task):
            bonus += self.following_dependencies_bonus

        # Cross-referencing multiple data sources
        if self._is_cross_referencing(action_type, state):
            bonus += self.cross_referencing_bonus

        # Ruling out red herrings (services that are not in causal chain)
        if self._is_ruling_out_red_herrings(action_type, state, task):
            bonus += self.ruling_out_red_herrings_bonus

        return bonus

    def _reward_penalties(self, action_type: str, state: IncidentState, task: IncidentTask) -> float:
        """Apply penalties for poor investigation strategies."""
        penalty = 0.0

        # Redundant queries
        if self._is_redundant_query(action_type, state):
            penalty += self.redundant_queries_penalty

        # Chasing red herrings
        if self._is_chasing_red_herrings(action_type, state, task):
            penalty += self.chasing_red_herrings_penalty

        # Circular investigation
        if self._is_circular_investigation(action_type, state):
            penalty += self.circular_investigation_penalty

        return penalty

    def _evaluate_diagnosis(self, state: IncidentState, task: IncidentTask) -> tuple[float, bool]:
        """Evaluate submitted diagnosis and return terminal reward."""
        if not hasattr(state, 'submitted_diagnosis') or not state.submitted_diagnosis:
            return 0.0, False

        diagnosis = state.submitted_diagnosis
        reward = 0.0
        done = True

        # Service accuracy
        if diagnosis.get('service') == task.root_cause_service:
            reward += 0.20
        elif diagnosis.get('service') in task.dependencies.get(task.root_cause_service, []):
            reward += 0.10  # One hop away

        # Category accuracy
        if diagnosis.get('category') == task.root_cause_category.value:
            reward += 0.15
        elif diagnosis.get('category') and diagnosis.get('category').split('_')[0] == str(task.root_cause_category.value).split('_')[0]:
            reward += 0.07  # Same failure family

        # Remediation accuracy
        if diagnosis.get('remediation') == task.correct_remediation.value:
            reward += 0.10
        elif diagnosis.get('remediation') and any(rem in str(diagnosis.get('remediation')).lower() for rem in str(task.correct_remediation.value).lower().split('_')):
            reward += 0.05  # Partial credit

        # Evidence quality - % of causal chain investigated
        causal_chain = self._get_causal_chain(task)
        investigated_services = self._get_investigated_services(state)
        evidence_score = min(0.10, 0.10 * (len(investigated_services.intersection(causal_chain)) / len(causal_chain)))
        reward += evidence_score

        # Efficiency bonus
        efficiency_ratio = max(0, 1 - (state.step_count / task.budget))
        reward += 0.15 * efficiency_ratio

        return reward, done

    def _is_redundant_query(self, info_type: str, state: IncidentState) -> bool:
        """Check if this query type has been done recently."""
        recent_actions = state.actions_taken[-3:]  # Last 3 actions
        query_actions = {
            "logs": "query_logs",
            "metrics": "query_metrics",
            "deploys": "check_deploys",
            "status": "check_status",
            "code": "inspect_code",
            "dependencies": "trace_dependencies",
        }
        action_type = query_actions.get(info_type)
        return recent_actions.count(action_type) > 1

    def _is_following_dependencies(self, action_type: str, state: IncidentState, task: IncidentTask) -> bool:
        """Check if agent is following dependency relationships."""
        if action_type not in ["query_logs", "query_metrics", "check_deploys", "inspect_code"]:
            return False

        # Check if the service being investigated is a dependency of a previously investigated service
        last_services = getattr(state, 'recently_investigated_services', [])
        current_service = getattr(state, 'last_action_params', {}).get('service_name')

        if not current_service or not last_services:
            return False

        for last_service in last_services[-2:]:  # Check last 2 services
            if current_service in task.dependencies.get(last_service, []):
                return True

        return False

    def _is_cross_referencing(self, action_type: str, state: IncidentState) -> bool:
        """Check if agent is cross-referencing multiple data sources for same service."""
        if action_type not in ["query_logs", "query_metrics", "check_deploys", "inspect_code"]:
            return False

        current_service = getattr(state, 'last_action_params', {}).get('service_name')
        if not current_service:
            return False

        # Count different investigation types for this service
        service_investigations = getattr(state, 'service_investigation_count', {}).get(current_service, {})
        return len(service_investigations) >= 2

    def _is_ruling_out_red_herrings(self, action_type: str, state: IncidentState, task: IncidentTask) -> bool:
        """Check if agent is investigating services that rule out red herrings."""
        current_service = getattr(state, 'last_action_params', {}).get('service_name')
        if not current_service:
            return False

        causal_chain = self._get_causal_chain(task)
        return current_service not in causal_chain and action_type in ["query_logs", "query_metrics"]

    def _is_chasing_red_herrings(self, action_type: str, state: IncidentState, task: IncidentTask) -> bool:
        """Check if agent is spending too much time on non-causal services."""
        current_service = getattr(state, 'last_action_params', {}).get('service_name')
        if not current_service:
            return False

        causal_chain = self._get_causal_chain(task)
        if current_service in causal_chain:
            return False

        # Count investigations of this non-causal service
        service_investigations = getattr(state, 'service_investigation_count', {}).get(current_service, 0)
        return service_investigations > 2

    def _is_circular_investigation(self, action_type: str, state: IncidentState) -> bool:
        """Check for circular investigation patterns."""
        recent_actions = state.actions_taken[-6:]  # Last 6 actions
        if len(recent_actions) < 6:
            return False

        # Check for repeating patterns
        action_counts = {}
        for action in recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1

        return any(count >= 3 for count in action_counts.values())

    def _get_causal_chain(self, task: IncidentTask) -> set[str]:
        """Get all services in the causal chain from symptoms to root cause."""
        chain = {task.root_cause_service}
        current = task.root_cause_service

        # Walk up the dependency graph
        while current in task.dependencies:
            for dep in task.dependencies[current]:
                if dep not in chain:
                    chain.add(dep)
                    current = dep
                    break
            else:
                break

        return chain

    def _get_investigated_services(self, state: IncidentState) -> set[str]:
        """Get set of services that have been investigated."""
        investigated = set()
        for action in state.actions_taken:
            if action in ["query_logs", "query_metrics", "check_deploys", "inspect_code"]:
                # In a real implementation, we'd track service parameters per action
                # For now, assume some services were investigated
                pass
            # return investigated
            elif action == "restart_service":
                reward += self._reward_action_taking()
                state.status = IncidentStatus.INVESTIGATING
            elif action == "scale_service":
                reward += self._reward_action_taking() * 1.5
                state.status = IncidentStatus.INVESTIGATING

            # Resolution actions
            elif action == "resolve_incident":
                if state.status == IncidentStatus.INVESTIGATING:
                    reward += self._reward_resolution()
                    state.status = IncidentStatus.RESOLVED
                else:
                    # Premature resolution attempt
                    reward -= 2.0
                done = True

            elif action == "escalate":
                reward += self._penalty_escalation()
                state.status = IncidentStatus.ESCALATED
                done = True

        # Inefficiency penalty (encourage quick resolution)
        reward += self._penalty_inefficiency(state.step_count)

        # Terminal condition: max steps reached
        if state.step_count >= self.max_steps:
            done = True
            if state.status != IncidentStatus.RESOLVED:
                reward += self._penalty_timeout()

        return reward, done

    def _reward_information_gathering(self) -> float:
        """Reward for diagnostic actions (logs, metrics, kb)."""
        return self.info_gathering_reward

    def _reward_action_taking(self) -> float:
        """Reward for mitigation actions (restart, scale)."""
        return self.investigation_reward

    def _reward_resolution(self) -> float:
        """Reward for successful resolution."""
        return self.resolution_reward

    def _penalty_escalation(self) -> float:
        """Penalty for escalation (negative outcome)."""
        return self.escalation_penalty

    def _penalty_inefficiency(self, step_count: int) -> float:
        """Penalize taking many steps."""
        return self.inefficiency_penalty_per_step * (step_count - 1)

    def _penalty_timeout(self) -> float:
        """Penalty for not resolving within max steps."""
        return -5.0
