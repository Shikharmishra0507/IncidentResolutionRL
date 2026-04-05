---
title: Incident Response RL Environment
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - incident-response
  - microservices
---

# Incident Response RL Environment

A reinforcement learning environment that simulates incident response scenarios in a microservices architecture. Agents learn to investigate service failures by gathering evidence, analyzing dependencies, and submitting accurate diagnoses within a step budget.

## Overview

This environment challenges RL agents to:
- **Investigate incidents**: Query logs, metrics, deployments, and service status
- **Analyze dependencies**: Trace service relationships and identify causal chains
- **Diagnose root causes**: Identify failing services, failure categories, and appropriate remediations
- **Optimize efficiency**: Complete investigations within step budgets while maximizing evidence quality

## Key Features

- **Realistic scenarios**: Multiple difficulty levels with service topologies from 3-8 services
- **Dense rewards**: Per-step learning signals for investigation strategies
- **Action space**: 8 actions including log queries, metric checks, code inspection, and diagnosis submission
- **Observation space**: Rich state including service topology, logs, metrics, and knowledge base results
- **Grading system**: Comprehensive evaluation of diagnosis accuracy, evidence quality, and efficiency

## Quick Start

### Using the Environment

```python
from rl_demo import IncidentEnv, IncidentAction

# Create environment instance
env = IncidentEnv()

# Reset for a new incident
observation = env.reset()
print(f"Incident: {observation.title}")
print(f"Description: {observation.description}")
print(f"Services: {[s['name'] for s in observation.services]}")

# Take investigation actions
action = IncidentAction(
    action_type="query_logs",
    parameters={"service_name": "api-service"}
)
result = env.step(action)
print(f"Reward: {result.reward}, Done: {result.done}")

# Submit diagnosis when ready
diagnosis_action = IncidentAction(
    action_type="submit_diagnosis",
    parameters={
        "service": "api-service",
        "category": "bad_deploy",
        "remediation": "rollback_deploy"
    }
)
final_result = env.step(diagnosis_action)
print(f"Final reward: {final_result.reward}")
```

### Running the Server

```bash
# Install dependencies
pip install -r requirements.txt

# Start the server
python -m rl_demo.server.app

# Server runs on http://localhost:8000
```

### Docker Usage

```bash
# Build the image
docker build -t incident-response-env .

# Run the container
docker run -p 8000:8000 incident-response-env
```

## Environment Details

### Actions
- `query_logs`: Retrieve logs for a specific service
- `query_metrics`: Get performance metrics for a service
- `check_deploys`: View recent deployments for a service
- `trace_dependencies`: Analyze service dependency relationships
- `check_status`: Get health status of all services
- `inspect_code`: Review code inspection results for a service
- `search_kb`: Query knowledge base for relevant articles
- `submit_diagnosis`: Submit final diagnosis with service, category, and remediation

### Observations
- Incident metadata (ID, title, description, severity)
- Service topology and dependencies
- Logs, metrics, deployment history
- Status information and code inspection results
- Knowledge base search results
- Action history and step count

### Reward Structure
- **Information gathering**: Rewards for relevant evidence collection
- **Strategy bonuses**: Bonuses for following dependencies and cross-referencing
- **Penalties**: Deductions for redundant queries and poor investigation patterns
- **Terminal rewards**: Large rewards/penalties for correct/incorrect diagnoses
- **Efficiency**: Bonuses for completing tasks with fewer steps

## Task Scenarios

The environment includes predefined incident scenarios:

1. **Easy**: Bad deploy causing NullPointerException (3 services, 10 steps)
2. **Medium**: Config change cascading through dependencies (5 services, 15 steps)
3. **Hard**: Memory leak with red herrings (8 services, 20 steps)

Each task includes ground truth for automated grading and reward calculation.

## Architecture

- **Client** (`client.py`): OpenEnv-compatible wrapper for agent interaction
- **Server** (`server/`): FastAPI-based environment server
  - `app.py`: Server entry point
  - `incident_environment.py`: Core environment logic
  - `incident_tasks.py`: Task definitions and grading
  - `reward_function.py`: Dense reward computation
- **Models** (`models.py`): Pydantic schemas for actions, observations, and state
- **Inference** (`inference.py`): Example script for running trained agents

## Dependencies

- Python 3.8+
- OpenEnv framework
- FastAPI
- Pydantic
- Uvicorn (for server)

## Contributing

This environment is designed for research in autonomous incident response and RL applications in DevOps scenarios. Feel free to extend with additional failure modes, larger topologies, or custom reward functions.
- Connecting to the environment
- Container cleanup when you call `close()`

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t rl_demo-env:latest -f server/Dockerfile .
```

## Deploying to Hugging Face Spaces

You can easily deploy your OpenEnv environment to Hugging Face Spaces using the `openenv push` command:

```bash
# From the environment directory (where openenv.yaml is located)
openenv push

# Or specify options
openenv push --namespace my-org --private
```

The `openenv push` command will:
1. Validate that the directory is an OpenEnv environment (checks for `openenv.yaml`)
2. Prepare a custom build for Hugging Face Docker space (enables web interface)
3. Upload to Hugging Face (ensuring you're logged in)

### Prerequisites

- Authenticate with Hugging Face: The command will prompt for login if not already authenticated

### Options

- `--directory`, `-d`: Directory containing the OpenEnv environment (defaults to current directory)
- `--repo-id`, `-r`: Repository ID in format 'username/repo-name' (defaults to 'username/env-name' from openenv.yaml)
- `--base-image`, `-b`: Base Docker image to use (overrides Dockerfile FROM)
- `--private`: Deploy the space as private (default: public)

### Examples

```bash
# Push to your personal namespace (defaults to username/env-name from openenv.yaml)
openenv push

# Push to a specific repository
openenv push --repo-id my-org/my-env

# Push with a custom base image
openenv push --base-image ghcr.io/meta-pytorch/openenv-base:latest

# Push as a private space
openenv push --private

# Combine options
openenv push --repo-id my-org/my-env --base-image custom-base:latest --private
```

After deployment, your space will be available at:
`https://huggingface.co/spaces/<repo-id>`

The deployed space includes:
- **Web Interface** at `/web` - Interactive UI for exploring the environment
- **API Documentation** at `/docs` - Full OpenAPI/Swagger interface
- **Health Check** at `/health` - Container health monitoring
- **WebSocket** at `/ws` - Persistent session endpoint for low-latency interactions

## Environment Details

### Action
**RlDemoAction**: Contains a single field
- `message` (str) - The message to echo back

### Observation
**RlDemoObservation**: Contains the echo response and metadata
- `echoed_message` (str) - The message echoed back
- `message_length` (int) - Length of the message
- `reward` (float) - Reward based on message length (length × 0.1)
- `done` (bool) - Always False for echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The reward is calculated as: `message_length × 0.1`
- "Hi" → reward: 0.2
- "Hello, World!" → reward: 1.3
- Empty message → reward: 0.0

## Advanced Usage

### Connecting to an Existing Server

If you already have a Rl Demo environment server running, you can connect directly:

```python
from rl_demo import RlDemoEnv

# Connect to existing server
rl_demoenv = RlDemoEnv(base_url="<ENV_HTTP_URL_HERE>")

# Use as normal
result = rl_demoenv.reset()
result = rl_demoenv.step(RlDemoAction(message="Hello!"))
```

Note: When connecting to an existing server, `rl_demoenv.close()` will NOT stop the server.

### Using the Context Manager

The client supports context manager usage for automatic connection management:

```python
from rl_demo import RlDemoAction, RlDemoEnv

# Connect with context manager (auto-connects and closes)
with RlDemoEnv(base_url="http://localhost:8000") as env:
    result = env.reset()
    print(f"Reset: {result.observation.echoed_message}")
    # Multiple steps with low latency
    for msg in ["Hello", "World", "!"]:
        result = env.step(RlDemoAction(message=msg))
        print(f"Echoed: {result.observation.echoed_message}")
```

The client uses WebSocket connections for:
- **Lower latency**: No HTTP connection overhead per request
- **Persistent session**: Server maintains your environment state
- **Efficient for episodes**: Better for many sequential steps

### Concurrent WebSocket Sessions

The server supports multiple concurrent WebSocket connections. To enable this,
modify `server/app.py` to use factory mode:

```python
# In server/app.py - use factory mode for concurrent sessions
app = create_app(
    RlDemoEnvironment,  # Pass class, not instance
    RlDemoAction,
    RlDemoObservation,
    max_concurrent_envs=4,  # Allow 4 concurrent sessions
)
```

Then multiple clients can connect simultaneously:

```python
from rl_demo import RlDemoAction, RlDemoEnv
from concurrent.futures import ThreadPoolExecutor

def run_episode(client_id: int):
    with RlDemoEnv(base_url="http://localhost:8000") as env:
        result = env.reset()
        for i in range(10):
            result = env.step(RlDemoAction(message=f"Client {client_id}, step {i}"))
        return client_id, result.observation.message_length

# Run 4 episodes concurrently
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(run_episode, range(4)))
```

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 server/rl_demo_environment.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running Locally

Run the server locally for development:

```bash
uvicorn server.app:app --reload
```

## Project Structure

```
rl_demo/
├── .dockerignore         # Docker build exclusions
├── __init__.py            # Module exports
├── README.md              # This file
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml         # Project metadata and dependencies
├── uv.lock                # Locked dependencies (generated)
├── client.py              # RlDemoEnv client
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── rl_demo_environment.py  # Core environment logic
    ├── app.py             # FastAPI application (HTTP + WebSocket endpoints)
    └── Dockerfile         # Container image definition
```
