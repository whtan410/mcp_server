# routers/sketchfab_router.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Mapping, Optional, Dict, Any
 
# services/sketchfab_service.py
from langchain.llms.base import LLM
import os
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
#from langchain_community.llms.dashscope import DashScope # Import Qwen model
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

load_dotenv(override=True)

router = APIRouter(prefix="/api/sketchfabqwen", tags=["sketchfabqwen"])

sketch_fab_api_key = os.getenv("SKETCHFAB_API_KEY")
dashscope_api_key = os.getenv("DASH_SCOPE_API_KEY")
index_js = os.getenv("MCP_BUILD_INDEX")

server_params = StdioServerParameters(
    command="node",
    args=[index_js, "--api-key", sketch_fab_api_key],
)

# Use OpenAI-compatible API to access Qwen
model = ChatOpenAI(
    model="qwen-plus",  # Or qwen-max, qwen-turbo, etc.
    api_key=os.getenv("DASH_SCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    max_tokens=4096,  # Adjust based on your needs
)


system_instructions =  """
        You have access to multiple tools, including one to search Sketchfab for 3D models and retrieve specific model details. Use tools sequentially when needed. Think step by step.

        IMPORTANT RULE: When a user asks a question in the format "What is [THING]?" or "Define [THING]" or any similar query asking for an explanation of [THING], you MUST perform the following actions and then respond with a specific JSON object. This JSON object should be the ONLY content in your response.

        **Procedure:**
        1.  **Define [THING]:** First, formulate a concise explanation or definition of [THING].
        2.  **Find Sketchfab Model via Tool:**
            *   Use your Sketchfab tool to search for [THING].
            *   Attempt to identify the "highest-rated" or most relevant model related to [THING].
            *   From the tool's results for this model, extract:
                *   The model's exact title.
                *   The model's direct preview URL (e.g., `https://sketchfab.com/models/xxxxxxxxxxxxxxxxx`).
                *   Whether the model is downloadable (true/false).
            *   If your tool cannot find a specific model for [THING], or if rating information isn't available to determine "highest-rated," try to pick the most relevant one. If no model is found at all, the model-specific fields in the JSON should be `null` or an empty string.
        3.  **Construct JSON Response:**
            The JSON object MUST have the following top-level keys:
            *   `"definition"`: The value should be the explanation of [THING] from step 1.
            *   `"model_name"`: The value should be the title of the Sketchfab model found (e.g., "Smart Drone"). Use `null` or an empty string if no model is found/applicable.
            *   `"preview_link"`: The value should be the direct Sketchfab preview URL for the *specific model* found (e.g., "https://sketchfab.com/models/76430ca6692b4238a54b5d12e90550c5"). Use `null` or an empty string if no model is found/applicable.
            *   `"is_downloadable"`: A boolean (`true` or `false`) indicating if the model is downloadable. Use `null` if this information isn't available or no model is found.

        **Example Interaction (assuming the tool found "Smart Drone" for "drone"):**
        User: What is a drone?
        Assistant (Your response):
        ```json
        {
            "definition": "A drone, also known as an unmanned aerial vehicle (UAV), is an aircraft without a human pilot aboard. Its flight is controlled either autonomously by onboard computers or by the remote control of a pilot on the ground or in another vehicle.",
            "model_name": "Smart Drone",
            "preview_link": "https://sketchfab.com/models/76430ca6692b4238a54b5d12e90550c5",
            "is_downloadable": false
        }
        ```

        **Another Example (assuming a relevant "neural network" model was found by the tool):**
        User: what is a neural network
        Assistant (Your response):
        ```json
        {
        "definition": "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. It is a subset of machine learning and is at the heart of deep learning algorithms.",
        "model_name": "Artificial Neural Network Visualization",
        "preview_link": "https://sketchfab.com/models/exampleModelIdForNeuralNet",
        "is_downloadable": true
        }
        ```

        **Example (if no relevant model is found by the tool for "love"):**
        User: What is love?
        Assistant (Your response):
        ```json
        {
        "definition": "Love encompasses a range of strong and positive emotional and mental states, from the most sublime virtue or good habit, the deepest interpersonal affection, to the simplest pleasure.",
        "model_name": null,
        "preview_link": null,
        "is_downloadable": null
        }
        ```
        Always adhere strictly to this procedure and JSON output format for such definition/explanation requests. Do not include any other text before or after the JSON object.
        """

@router.post("/model-info")
async def get_model_info(query: str) -> Dict[str, Any]:
    """Process a query and return model information as a JSON-serializable dict"""
    try:
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(session)
                agent = create_react_agent(model=model, tools=tools, prompt=system_instructions)
                attempt = 0
                search_tool_used = False
                
                while attempt < 5 and not search_tool_used:
                    res = await agent.ainvoke({"messages": [("human", query)]})
                    print(res)
                    if len(res["messages"]) >= 2 and (res["messages"][-2].type == "tool" or res["messages"][-2].type == "ai"):
                        search_tool_used = True
                    else:
                        attempt += 1

                print(res)
                final_result = res["messages"][-1].content
                print(f"DEBUG: Raw result: {final_result}")
                
                # Check if the result is wrapped in markdown code block
                if isinstance(final_result, str) and "```json" in final_result:
                    # Extract just the JSON part
                    import re
                    import json
                    # Match anything between ```json and ``` using regex
                    match = re.search(r'```json\s*([\s\S]*?)\s*```', final_result)
                    if match:
                        json_str = match.group(1).strip()
                        print(f"DEBUG: Extracted JSON string: {json_str}")
                        # Parse the JSON string into a Python dict
                        final_result = json.loads(json_str)
                        print(f"DEBUG: Parsed result: {final_result}")
                
                return {"response": final_result}

    except Exception as e:
        print(e)
        return {"error": str(e)}