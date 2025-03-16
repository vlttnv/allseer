import asyncio
import os
import time
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich import print
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from utils import configure_logging, get_logger

logger = get_logger(__name__)

# Create prompt object.
prompt_session = PromptSession()
console = Console()


class TokenCounter:
    """Class to track token usage and calculate costs for Claude API."""

    PRICING = {"input": 3.0, "output": 15.0}

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.request_count = 0
        self.tool_call_count = 0
        self.session_start_time = time.time()

    def track_request(self, response: Any) -> None:
        """Track tokens for a request to the Claude API using values from response."""
        if hasattr(response, "usage") and response.usage:
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
            self.request_count += 1
            logger.info(
                f"Request used {response.usage.input_tokens} input tokens, {response.usage.output_tokens} output tokens"
            )
        else:
            # Fallback if usage information is not available
            logger.warning("Token usage information not available in response")
            self.request_count += 1

    def track_tool_call(self):
        """Track a tool call."""
        self.tool_call_count += 1

    def calculate_cost(self) -> Dict[str, float]:
        """Calculate the cost based on token usage."""
        input_cost = (self.input_tokens / 1_000_000) * self.PRICING["input"]
        output_cost = (self.output_tokens / 1_000_000) * self.PRICING["output"]
        total_cost = input_cost + output_cost

        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get all token usage statistics."""
        costs = self.calculate_cost()
        session_duration = time.time() - self.session_start_time

        return {
            "request_count": self.request_count,
            "tool_call_count": self.tool_call_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "input_cost": costs["input_cost"],
            "output_cost": costs["output_cost"],
            "total_cost": costs["total_cost"],
            "session_duration": session_duration,
        }

    def display_stats(self):
        """Display token usage statistics in a nice format."""
        stats = self.get_stats()

        table = Table(title="Claude API Usage Statistics")

        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("API Requests", str(stats["request_count"]))
        table.add_row("Tool Calls", str(stats["tool_call_count"]))
        table.add_row("Input Tokens", f"{stats['input_tokens']:,}")
        table.add_row("Output Tokens", f"{stats['output_tokens']:,}")
        table.add_row("Total Tokens", f"{stats['total_tokens']:,}")
        table.add_row("Input Cost", f"${stats['input_cost']:.4f}")
        table.add_row("Output Cost", f"${stats['output_cost']:.4f}")
        table.add_row("Total Cost", f"${stats['total_cost']:.4f}")
        table.add_row("Session Duration", f"{stats['session_duration']:.1f} seconds")

        console.print(table)


class MCPClient:
    MODEL = "claude-3-7-sonnet-20250219"

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

        # Initialize conversation history
        self.messages = []
        self.conversation_started = False

        # Initialize token counter
        self.token_counter = TokenCounter()

        # Load system prompt from file
        prompt_path = os.path.join(os.path.dirname(__file__), "system_prompt.md")
        try:
            with open(prompt_path, "r") as f:
                self.SYSTEM_PROMPT = f.read()
        except FileNotFoundError:
            logger.error(f"System prompt file not found at {prompt_path}")
            # Fallback to a basic prompt if file not found
            self.SYSTEM_PROMPT = "You are Allseer, an expert DevOps Engineer and SRE specializing in Kubernetes. Help troubleshoot issues and provide actionable solutions."

        # Configure logging
        configure_logging("ERROR", library_loggers={"mcp": "ERROR"})

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=os.environ
        )

        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(self.stdio, self.write)
        )

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logger.info("\nConnected to server with tools:", [tool.name for tool in tools])

    async def process_query(self, query: str):
        """Process a query using Claude and available tools"""
        self.messages.append({"role": "user", "content": query})
        # final_text = []

        # Get available tools
        response = await self.session.list_tools()
        available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in response.tools
        ]

        while True:
            # Call Claude API with system prompt only on first message
            system = self.SYSTEM_PROMPT if not self.conversation_started else []
            self.conversation_started = True

            response = self.anthropic.messages.create(
                model=self.MODEL,
                max_tokens=1000,
                messages=self.messages,
                system=system,
                tools=available_tools,
            )

            # Track tokens from the API response
            self.token_counter.track_request(response)

            has_tool_calls = False
            assistant_message_content = []

            # Process each content block in the response
            for content in response.content:
                if content.type == "text":
                    print(Markdown(content.text))

                    assistant_message_content.append(
                        {"type": "text", "text": content.text}
                    )
                elif content.type == "tool_use":
                    has_tool_calls = True
                    self.token_counter.track_tool_call()

                    tool_name = content.name
                    tool_args = content.input

                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)

                    # Add to final text
                    print(
                        Markdown(
                            f"[Calling tool `{tool_name}` with args `{tool_args}`]"
                        )
                    )

                    # Add the tool use to the conversation
                    assistant_message_content.append(
                        {
                            "type": "tool_use",
                            "name": tool_name,
                            "input": tool_args,
                            "id": content.id,
                        }
                    )

                    # Add assistant's message to the conversation
                    self.messages.append(
                        {"role": "assistant", "content": assistant_message_content}
                    )

                    # Add tool response to the conversation
                    self.messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": content.id,
                                    "content": result.content,
                                }
                            ],
                        }
                    )

                    # Reset assistant message content for potential next tool calls
                    assistant_message_content = []

                    # Break out of the current response processing to get Claude's next response
                    break

            # If no tool calls or we've processed all content, add the assistant message and exit
            if not has_tool_calls:
                if assistant_message_content:
                    self.messages.append(
                        {"role": "assistant", "content": assistant_message_content}
                    )
                break

        return

    async def chat_loop(self):
        """Run an interactive chat loop"""
        self._print_logo()
        print("\nAlsseer: an AI-powered diagnostics assistant for Kubernetes.")
        print("Type your queries, 'stats' to see token usage, or 'quit' to exit.")

        while True:
            try:
                with patch_stdout():
                    query = await prompt_session.prompt_async("> ")

                if query.lower() in ["quit", "exit"]:
                    # Display final stats before exiting
                    print("\nFinal Usage Statistics:")
                    self.token_counter.display_stats()
                    break

                elif query.lower() == "stats":
                    # Display current stats without exiting
                    self.token_counter.display_stats()
                    continue

                await self.process_query(query)
                # Report current token usage after each interaction in logs
                logger.info(
                    f"Session token usage so far: input={self.token_counter.input_tokens}, output={self.token_counter.output_tokens}"
                )

            except Exception as e:
                logger.error(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    def _print_logo(self):
        print("   _____  .__  .__                              ")
        print("  /  _  \ |  | |  |   ______ ____   ___________ ")
        print(" /  /_\  \|  | |  |  /  ___// __ \_/ __ \_  __ \\")
        print("/    |    \  |_|  |__\___ \\\\  ___/\  ___/|  | \/")
        print("\____|__  /____/____/____  >\___  >\___  >__|   ")
        print("        \/               \/     \/     \/       ")


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    import sys

    asyncio.run(main())
