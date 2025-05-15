# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv

load_dotenv()



from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["server.py"],
)

llm = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2025-01-01-preview",
            temperature=0.1,
            max_tokens=None,
        )

async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)

            # Create and run the agent
            agent = create_react_agent(llm, tools)
            final_state = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})

            print("\n--- AGENT INTERACTION LOG ---")
            for i, msg in enumerate(final_state["messages"]):
                if msg.type == "human":
                    print(f"Step {i+1}: User asked: \"{msg.content}\"")
                elif msg.type == "ai":
                    if msg.tool_calls:
                        print(f"Step {i+1}: AI decided to use tool(s):")
                        for tc in msg.tool_calls:
                            # tc is a dict: {'name': 'tool_name', 'args': {'arg1': 'val1'}, 'id': 'call_id'}
                            print(f"  - Tool: {tc['name']}, Args: {tc['args']}, ID: {tc['id']}")
                    else:
                        # This is likely the final answer from the AI
                        print(f"Step {i+1}: AI responded: \"{msg.content}\"")
                elif msg.type == "tool":
                    # msg is a ToolMessage: content='result', name='tool_name', tool_call_id='call_id'
                    print(f"Step {i+1}: Tool '{msg.name}' (ID: {msg.tool_call_id}) executed and returned: \"{msg.content}\"")
            print("--- END OF AGENT INTERACTION LOG ---\n")

import asyncio
asyncio.run(main())
