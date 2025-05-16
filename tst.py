from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()

print(os.getenv("AZURE_OPENAI_API_KEY"))
print(os.getenv("AZURE_OPENAI_ENDPOINT"))
print(os.getenv("AZURE_API_VERSION"))

def test_azure_openai():
    try:
        # Initialize the model
        model = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2025-01-01-preview",
            temperature=0.1,
            max_tokens=None,
        )
        
        # Generate a simple response
        response = model.invoke("Generate a short poem about learning")
        
        # Display the response
        print("API CONNECTION SUCCESSFUL!")
        print("Model response:")
        print(response.content)
        return True
        
    except Exception as e:
        print("API CONNECTION FAILED!")
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    test_azure_openai()