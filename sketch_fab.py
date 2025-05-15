from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

sketch_fab_api_key = os.getenv("SKETCHFAB_API_KEY")

model= AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2025-01-01-preview",
            temperature=0.1,
            max_tokens=None,
        )

server_params = StdioServerParameters(
    command="node",
    args=["C:/Users/tanweihan/Documents/GitHub/sketchfab-mcp-server/build/index.js", "--api-key", sketch_fab_api_key],
)

async def chat_with_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            agent = create_react_agent(model, tools)

            # Start conversation history
            messages = [
                {
                    "role": "system",
                    "content": "You can use multiple tools in sequence to answer complex questions. Think step by step.",
                }
            ]

            print("Type 'exit' or 'quit' to end the chat.")
            while True:
                user_input = input("\nYou: ")
                if user_input.strip().lower() in {"exit", "quit"}:
                    print("Goodbye!")
                    break

                # Add user message to history
                messages.append({"role": "user", "content": user_input})

                # Call the agent with the full message history
                agent_response = await agent.ainvoke({"messages": messages})

                # Extract agent's reply and add to history
                ai_message = agent_response["messages"][-1].content
                print(f"Agent: {ai_message}")


if __name__ == "__main__":
    asyncio.run(chat_with_agent())

