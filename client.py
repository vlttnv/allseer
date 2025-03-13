import asyncio
from contextlib import AsyncExitStack
from typing import Optional

from anthropic import Anthropic
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from utils import configure_logging, get_logger

from rich import print
from rich.markdown import Markdown

import os
from prompt_toolkit import PromptSession
from prompt_toolkit.patch_stdout import patch_stdout

logger = get_logger(__name__)

# Create prompt object.
prompt_session = PromptSession()


class MCPClient:
    MODEL = "claude-3-7-sonnet-20250219"
    SYSTEM_PROMPT = (
        "You are Allseer, an expert DevOps Engineer and SRE, specializing in Kubernetes. Your mission is to help experienced DevOps and SRE engineers troubleshoot issues, surface insights, and save time by analyzing cluster data and connecting the dots across cluster resources.\n"
        "- **Behavior:** Be concise, technical, and actionable. Respond like a seasoned engineer giving a clear, no-nonsense explanation. Avoid fluff. Focus on facts, root causes, and fixes.\n"
        "- **Tone:** Professional, confident, and slightly informal — like a trusted colleague. Use markdown for structured output (e.g., headings, code blocks).\n"
        "- **Constraints:** You cannot modify the cluster — only suggest fixes (e.g., commands, YAML) for the user to apply. If data is missing, say so and suggest what's needed.\n"
        "- **Output Format:** When diagnosing, structure responses with: 1) Problem Analysis (what's happening), 2) Root Cause (why it's happening), 3) How to Fix (actionable steps).\n"
    )

    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.anthropic = Anthropic()

        # Initialize conversation history
        self.messages = []

        # Configure logging
        configure_logging("ERROR")

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

    async def process_query(self, query: str) -> str:
        """Process a query using Claude and available tools"""
        # messages = [{"role": "user", "content": query}]
        self.messages.append({"role": "user", "content": query})
        final_text = []

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
            # Call Claude API
            response = self.anthropic.messages.create(
                model=self.MODEL,
                max_tokens=1000,
                messages=self.messages,
                system=self.SYSTEM_PROMPT,
                tools=available_tools,
            )

            has_tool_calls = False
            assistant_message_content = []

            # Process each content block in the response
            for content in response.content:
                if content.type == "text":
                    print(Markdown(content.text))
                    final_text.append(content.text)
                    assistant_message_content.append({"type": "text", "text": content.text})
                elif content.type == "tool_use":
                    has_tool_calls = True
                    tool_name = content.name
                    tool_args = content.input

                    # Execute tool call
                    result = await self.session.call_tool(tool_name, tool_args)

                    # Add to final text
                    print(Markdown(f"[Calling tool `{tool_name}` with args `{tool_args}`]"))
                    final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                    final_text.append(f"[Tool result: {result.content[:100]}...]")

                    # Add the tool use to the conversation
                    assistant_message_content.append({
                        "type": "tool_use",
                        "name": tool_name,
                        "input": tool_args,
                        "id": content.id
                    })

                    # Add assistant's message to the conversation
                    self.messages.append({"role": "assistant", "content": assistant_message_content})

                    # Add tool response to the conversation
                    self.messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content
                            }
                        ]
                    })

                    # Reset assistant message content for potential next tool calls
                    assistant_message_content = []

                    # Break out of the current response processing to get Claude's next response
                    break

            # If no tool calls or we've processed all content, add the assistant message and exit
            if not has_tool_calls:
                if assistant_message_content:
                    self.messages.append({"role": "assistant", "content": assistant_message_content})
                break

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        self._print_logo()
        print("\nAlsseer: an AI-powered diagnostics assistant for Kubernetes.")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                with patch_stdout():
                    query = await prompt_session.prompt_async("> ")

                if query.lower() in ["quit", "exit"]:
                    break

                response = await self.process_query(query)
                # md = Markdown(response)
                # print(md)

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
