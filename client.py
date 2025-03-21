import asyncio
import os
import time
import json
from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

from anthropic import AsyncAnthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout
from rich import print
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

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
        self.anthropic = AsyncAnthropic()

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
        """Process a query using Claude and available tools with streaming responses"""
        self.messages.append({"role": "user", "content": query})

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

        # Process through multiple API calls if tools are used
        while True:
            # Set system prompt only on first message
            system = self.SYSTEM_PROMPT if not self.conversation_started else ""
            self.conversation_started = True

            # Initialize collection variables
            assistant_message_content = []
            has_tool_calls = False
            current_text_chunk = ""

            # Use Rich's Live display for updating content in-place
            with Live(
                Panel("", title="Allseer", title_align="left", border_style="blue", expand=False),
                console=console,
                refresh_per_second=10,
                transient=False
            ) as live_display:
                # Create a streaming response using AsyncAnthropic with stream context manager
                async with self.anthropic.messages.stream(
                    model=self.MODEL,
                    max_tokens=1000,
                    messages=self.messages,
                    system=system,
                    tools=available_tools,
                ) as stream:
                    try:
                        # Process each event in the stream
                        async for event in stream:
                            if event.type == "message_start":
                                # Track tokens from the API response
                                if hasattr(event, "message") and hasattr(event.message, "usage"):
                                    self.token_counter.track_request(event.message)

                            elif event.type == "content_block_start":
                                if event.content_block.type == "text":
                                    current_text_chunk = ""

                            elif event.type == "text":
                                # Build the text as it streams in
                                current_text_chunk += event.text
                                # Update the live display with new text
                                live_display.update(
                                    Panel(Markdown(current_text_chunk), title="Allseer", title_align="left", border_style="blue")
                                )

                            elif event.type == "content_block_stop":
                                if event.content_block.type == "text":
                                    # Add final text block to message content
                                    assistant_message_content.append({
                                        "type": "text",
                                        "text": current_text_chunk
                                    })

                                elif event.content_block.type == "tool_use":
                                    has_tool_calls = True
                                    tool_block = event.content_block
                                    self.token_counter.track_tool_call()

                                    # Display tool call information in the console (outside the live display)
                                    live_display.stop()
                                    console.print(Panel(
                                        f"Tool: {tool_block.name}\nInput: {tool_block.input}",
                                        title="Tool Call",
                                        title_align="left",
                                        border_style="yellow"
                                    ))

                                    # Add the tool use to the message content
                                    assistant_message_content.append({
                                        "type": "tool_use",
                                        "name": tool_block.name,
                                        "input": tool_block.input,
                                        "id": tool_block.id,
                                    })

                                    # Execute tool call
                                    result = await self.session.call_tool(tool_block.name, tool_block.input)

                                    # Display tool result in a panel with proper formatting
                                    try:
                                        # Try to format as nice JSON if possible
                                        formatted_result = json.dumps(json.loads(result.content[0].text), indent=2)
                                        console.print(Panel(
                                            Markdown(f"```json\n{formatted_result}\n```"),
                                            title="Tool Result",
                                            title_align="left",
                                            border_style="green"
                                        ))
                                    except:
                                        # Fallback to plain text if not valid JSON
                                        console.print(Panel(
                                            Markdown(result.content[0].text),
                                            title="Tool Result",
                                            title_align="left",
                                            border_style="green"
                                        ))

                                    # Add assistant's message to the conversation
                                    self.messages.append({
                                        "role": "assistant",
                                        "content": assistant_message_content
                                    })

                                    # Add tool response to the conversation
                                    self.messages.append({
                                        "role": "user",
                                        "content": [{
                                            "type": "tool_result",
                                            "tool_use_id": tool_block.id,
                                            "content": result.content,
                                        }],
                                    })

                                    # Exit the stream processing loop after handling a tool call
                                    # We'll start a new stream to get Claude's response to the tool result
                                    await stream.close()
                                    break

                            elif event.type == "message_stop":
                                # Get final message details for token tracking if available
                                if hasattr(event, "message") and hasattr(event.message, "usage"):
                                    self.token_counter.track_request(event.message)

                    except Exception as e:
                        logger.error(f"Error during streaming: {str(e)}")
                        live_display.stop()
                        console.print(Panel(
                            f"Error during streaming: {str(e)}",
                            title="Error",
                            title_align="left",
                            border_style="red"
                        ))
                        # Still track the request even if there was an error
                        self.token_counter.request_count += 1

            # If no tool calls or all content processed, add the assistant message and exit
            if not has_tool_calls:
                if assistant_message_content:
                    self.messages.append({
                        "role": "assistant",
                        "content": assistant_message_content
                    })
                break

            # Continue the loop to process Claude's response to the tool result

    async def chat_loop(self):
        """Run an interactive chat loop"""
        self._print_logo()
        console.print("\nAllseer: an AI-powered diagnostics assistant for Kubernetes.")
        console.print("Type your queries, 'stats' to see token usage, or 'quit' to exit.")

        while True:
            try:
                with patch_stdout():
                    query = await prompt_session.prompt_async("\n> ")

                if query.lower() in ["quit", "exit"]:
                    # Display final stats before exiting
                    console.print("\nFinal Usage Statistics:")
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
                console.print(Panel(
                    f"Error: {str(e)}",
                    title="Error",
                    title_align="left",
                    border_style="red"
                ))

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

    def _print_logo(self):
        logo = """   _____  .__  .__
  /  _  \ |  | |  |   ______ ____   ___________
 /  /_\  \|  | |  |  /  ___// __ \_/ __ \_  __ \\
/    |    \  |_|  |__\___ \\\\  ___/\  ___/|  | \/
\____|__  /____/____/____  >\___  >\___  >__|
        \/               \/     \/     \/       """

        console.print(Panel(Text(logo, style="bold blue"), border_style="blue"))


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
