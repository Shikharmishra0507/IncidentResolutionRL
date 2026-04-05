"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 rewards=0.00,0.00,1.00
"""

import asyncio
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from client import IncidentEnv
from models import IncidentAction
IMAGE_NAME = os.getenv("IMAGE_NAME") # If you are using docker image 
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("INCIDENT_TASK", "bad_deploy")
BENCHMARK = os.getenv("INCIDENT_BENCHMARK", "incident_response")
MAX_STEPS = 10  # Default for easy tasks
TEMPERATURE = 0.7
MAX_TOKENS = 150

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an incident response agent investigating system failures.
    Your goal is to identify the root cause service, failure category, and remediation.
    Use investigation actions to gather evidence, then submit your diagnosis.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(
        f"""
        Step: {step}
        Last echoed message: {last_echoed!r}
        Last reward: {last_reward:.2f}
        Previous steps:
        {history_block}
        Send your next message.
        """
    ).strip()


def get_incident_action(step: int, last_echoed: str, last_reward: float, history: List[str], observation) -> IncidentAction:
    """Simple strategy for selecting incident response actions."""
    # Cycle through investigation actions
    actions = [
        {"action_type": "query_logs", "parameters": {"service_name": "api-service"}},
        {"action_type": "query_metrics", "parameters": {"service_name": "api-service"}},
        {"action_type": "check_deploys", "parameters": {"service_name": "api-service"}},
        {"action_type": "check_status", "parameters": {}},
        {"action_type": "search_kb", "parameters": {"query": "performance issues"}},
        {"action_type": "inspect_code", "parameters": {"service_name": "api-service"}},
        {"action_type": "trace_dependencies", "parameters": {"service_name": "api-service"}},
    ]

    # Use step number to cycle through actions
    action_idx = (step - 1) % len(actions)
    action_data = actions[action_idx]

    # On final steps, submit a diagnosis
    if step >= MAX_STEPS - 1:
        action_data = {
            "action_type": "submit_diagnosis",
            "parameters": {
                "service": "api-service",  # Simple guess
                "category": "bad_deploy",
                "remediation": "rollback_deploy"
            }
        }

    return IncidentAction(**action_data)


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    # Create incident response environment
    env = IncidentEnv(base_url="http://localhost:8000")  # Server URL

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            # Simple action selection strategy for incident response
            action = get_incident_action(step, "", last_reward, history, result.observation)

            result = await env.step(action)
            obs = result.observation

            reward = result.reward or 0.0
            done = result.done
            error = None

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            # Log the action (convert to string representation)
            action_str = f"{action.action_type}"
            if action.parameters:
                action_str += f"({action.parameters})"

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append(f"Step {step}: {action_str} -> reward {reward:+.2f}")

            if done:
                break

        # Calculate score based on total rewards (simplified scoring)
        score = sum(rewards)
        success = score > 0.5  # Simple threshold

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    asyncio.run(main())