import sys
from uuid import uuid4

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from k8s_tools import (
    failed_pods,
    get_namespaces,
    get_resource_yaml,
    high_restart_pods,
    list_deployments,
    list_events,
    list_nodes,
    list_pods,
    list_services,
    node_capacity,
    orphaned_resources,
    pending_pods,
)


def create_agent():
    # Create the agent
    memory = MemorySaver()
    model = ChatAnthropic(model_name="claude-3-7-sonnet-latest")
    tools = [
        get_namespaces,
        list_pods,
        list_nodes,
        list_deployments,
        list_services,
        list_events,
        failed_pods,
        pending_pods,
        high_restart_pods,
        node_capacity,
        orphaned_resources,
        get_resource_yaml,
    ]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    return agent_executor


def run_chat_agent():
    agent_executor = create_agent()
    config = {"configurable": {"thread_id": uuid4()}}
    print(
        "Alsseer: an AI-powered diagnostics assistant for Kubernetes. Type 'quit' or 'Ctrl-C' to exit."
    )

    user_input = input("You: ")
    system_prompt = (
        "You are Allseer, an AI-powered diagnostics assistant for Kubernetes. Your mission is to help experienced DevOps and SRE engineers troubleshoot issues, surface insights, and save time by analyzing cluster data and connecting the dots across resources like pods, nodes, deployments, and events.\n"
        "- **Behavior:** Be concise, technical, and actionable. Respond like a seasoned engineer giving a clear, no-nonsense explanation. Avoid fluff. Focus on facts, root causes, and fixes.\n"
        "- **Tone:** Professional, confident, and slightly informal—like a trusted colleague. Use markdown for structured output (e.g., headings, code blocks).\n"
        "- **Constraints:** You cannot modify the cluster—only suggest fixes (e.g., commands, YAML) for the user to apply. If data is missing, say so and suggest what's needed.\n"
        "- **Output Format:** When diagnosing, structure responses with: 1) Problem Analysis (what's happening), 2) Root Cause (why it's happening), 3) How to Fix (actionable steps).\n"
    )

    for step in agent_executor.stream(
        {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_input),
            ]
        },
        config,
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()

    # Chat loop
    while True:
        # Get user input
        user_input = input("You: ")

        # Check for exit condition
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        for step in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values",
        ):
            step["messages"][-1].pretty_print()


# Main execution
if __name__ == "__main__":
    try:
        run_chat_agent()
    except KeyboardInterrupt:
        sys.exit(0)
