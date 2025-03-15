<div style="text-align: center;">
  <img src="./assets/logo.svg" alt="Allseer Logo" width="400" height="400" />
</div>

# Allseer
Allseer is read-only, AI-powered diagnostics assistant for Kubernetes that helps DevOps and SRE engineers troubleshoot issues, surface insights, and save time by analyzing cluster data and connecting the dots across resources.

## Installation

### Prerequisites

- [k8s-mpc](https://github.com/vlttnv/k8s-mcp) - cloned locally
- Python 3.8+
- Access to a Kubernetes cluster (via kubeconfig or in-cluster configuration)
- Required Python packages (see `dependencies` in `pyproject.toml`)
- uv - https://github.com/astral-sh/uv

```bash
# To install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
# Clone the repository
git clone git@github.com:vlttnv/allseer.git
cd allseer

# Install dependencies
uv venv
source .venv/bin/activate
uv sync
```

Currently only Claude is supported.
Export your Anthropic API key as an env variable like so:

```bash
export ANTHROPIC_API_KEY="YOUR_KEY"
```

## Configuration

The application automatically tries two methods to connect to your Kubernetes cluster:

1. **Kubeconfig File**: Uses your local kubeconfig file (typically located at `~/.kube/config`)
2. **In-Cluster Configuration**: If running inside a Kubernetes pod, uses the service account token

No additional configuration is required if your kubeconfig is properly set up or if you're running inside a cluster with appropriate RBAC permissions.

## Usage

### Examples
Here are some useful example prompts you can ask Claude about your Kubernetes cluster and its resources:

#### General Cluster Status
- "What's the overall health of my cluster?"
- "Show me all namespaces in my cluster"
- "What nodes are available in my cluster and what's their status?"
- "How is resource utilization across my nodes?"

#### Pods and Deployments
- "List all pods in the production namespace"
- "Are there any pods in CrashLoopBackOff state?"
- "Show me pods with high restart counts"
- "List all deployments across all namespaces"
- "What deployments are failing to progress?"

#### Debugging Issues
- "Why is my pod in the staging namespace failing?"
- "Get the YAML configuration for the service in the production namespace"
- "Show me recent events in the default namespace"
- "Are there any pods stuck in Pending state?"
- "What's causing ImagePullBackOff errors in my cluster?"

#### Resource Management
- "Show me the resource consumption of nodes in my cluster"
- "Are there any orphaned resources I should clean up?"
- "List all services in the production namespace"
- "Compare resource requests between staging and production"

#### Specific Resource Inspection
- "Show me the config for the coredns deployment in kube-system"
- "Get details of the reverse-proxy service in staging"
- "What containers are running in the pod xyz?"
- "Show me the logs for the failing pod"

## API Reference
The [k8s-mpc](https://github.com/vlttnv/k8s-mcp) server exposes a set of tools. For a full list check out the project's README or source code.

## License

[GNU GPLv3 License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
