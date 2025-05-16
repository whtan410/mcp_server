# services/sketchfab_service.py
import os
import asyncio
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import AzureChatOpenAI
from dotenv import load_dotenv
from typing import Dict, Any, Optional, List

load_dotenv()

class SketchfabService:
    def __init__(self):
        self.api_key = os.getenv("SKETCHFAB_API_KEY")
        self.model = AzureChatOpenAI(
            azure_deployment="gpt-4o",
            api_version="2025-01-01-preview",
            temperature=0.1,
            max_tokens=None,
        )
        self.system_instructions = [
            {
                "role": "system",
                "content": """You have access to multiple tools, including one to search Sketchfab for 3D models and retrieve specific model details. Use tools sequentially when needed. Think step by step.

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
                """
            }
        ]
        
    # async def get_model_info(self, query: str) -> Dict[str, Any]:
    #     """Process a query and return model information as a JSON-serializable dict"""
    #     try:
    #         server_params = StdioServerParameters(
    #             command="node",
    #             args=["C:/Users/pa662/Documents/GitHub/sketchfab-mcp-server/build/index.js", "--api-key", self.api_key],
    #         )
            
    #         async with stdio_client(server_params) as (read, write):
    #             async with ClientSession(read, write) as session:
    #                 await session.initialize()
    #                 tools = await load_mcp_tools(session)
    #                 agent = create_react_agent(model=self.model, tools=tools, prompt=self.system_instructions)
    #                 attempt = 0
    #                 search_tool_used = False
                    
    #                 while attempt < 5 and not search_tool_used:
    #                     res = await agent.ainvoke({"messages": [("human", query)]})
    #                     print(res)
    #                     if len(res["messages"]) >= 2 and (res["messages"][-2].type == "tool" or res["messages"][-2].type == "ai"):
    #                         search_tool_used = True
    #                     else:
    #                         attempt += 1

    #                 print(res)
    #                 final_result = res["messages"][-1].content
    #                 return final_result
    
    #     except Exception as e:
    #         print(e)
    #         return {"error": str(e)}

    async def get_model_info(self, query: str) -> Dict[str, Any]:
        """Process a query and return model information as a JSON-serializable dict"""
        try:
            print(f"DEBUG: Starting get_model_info with query: {query}")
            server_params = StdioServerParameters(
                command="node",
                args=["C:/Users/pa662/Documents/GitHub/sketchfab-mcp-server/build/index.js", "--api-key", self.api_key],
            )
            
            print(f"DEBUG: Server parameters set up with API key: {self.api_key[:5] if self.api_key else 'None'}...")
            async with stdio_client(server_params) as (read, write):
                print("DEBUG: Stdio client initialized")
                async with ClientSession(read, write) as session:
                    print("DEBUG: Client session started")
                    await session.initialize()
                    print("DEBUG: Session initialized")
                    tools = await load_mcp_tools(session)
                    print(f"DEBUG: MCP tools loaded: {[t.name for t in tools if hasattr(t, 'name')]}")
                    agent = create_react_agent(model=self.model, tools=tools, prompt=self.system_instructions)
                    print("DEBUG: React agent created")
                    attempt = 0
                    search_tool_used = False
                    
                    while attempt < 5 and not search_tool_used:
                        print(f"DEBUG: Attempt {attempt+1}/5 to get model info")
                        try:
                            res = await agent.ainvoke({"messages": [("human", query)]})
                            print(f"DEBUG: Agent response received, message count: {len(res['messages'])}")
                            print(f"DEBUG: Message types: {[m.type for m in res['messages'] if hasattr(m, 'type')]}")
                            
                            if len(res["messages"]) >= 2:
                                print(f"DEBUG: Checking message type: {res['messages'][-2].type}")
                                if res["messages"][-2].type == "tool" or res["messages"][-2].type == "ai":
                                    search_tool_used = True
                                    print("DEBUG: Search tool used successfully")
                                else:
                                    attempt += 1
                                    print(f"DEBUG: Search tool not used, incrementing attempt to {attempt}")
                            else:
                                attempt += 1
                                print(f"DEBUG: Not enough messages in response, incrementing attempt to {attempt}")
                        except Exception as inner_e:
                            print(f"DEBUG: Error in attempt {attempt+1}: {str(inner_e)}")
                            attempt += 1

                    print(f"DEBUG: Final response keys: {res.keys()}")
                    print(f"DEBUG: Final message count: {len(res['messages'])}")
                    final_result = res["messages"][-1].content
                    print(f"DEBUG: Final result type: {type(final_result)}")
                    
                    # Try to parse the result as JSON if it's a string
                    if isinstance(final_result, str):
                        try:
                            import json
                            parsed = json.loads(final_result)
                            print(f"DEBUG: Result parsed as JSON with keys: {list(parsed.keys())}")
                        except json.JSONDecodeError:
                            print("DEBUG: Result is not valid JSON")
                        except Exception as json_err:
                            print(f"DEBUG: Error parsing result: {str(json_err)}")
                    
                    return final_result

        except Exception as e:
            import traceback
            print(f"DEBUG ERROR: {str(e)}")
            print(f"DEBUG TRACEBACK: {traceback.format_exc()}")
            return {"error": str(e), "traceback": traceback.format_exc()}
        
    # async def search_models(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    #     """Search for models directly using the MCP tools"""
    #     try:
    #         server_params = StdioServerParameters(
    #             command="node",
    #             args=["C:/Users/pa662/Documents/GitHub/sketchfab-mcp-server/build/index.js", "--api-key", self.api_key],
    #         )
            
    #         async with stdio_client(server_params) as (read, write):
    #             async with ClientSession(read, write) as session:
    #                 await session.initialize()
    #                 result = await session.tool_call(
    #                     "sketchfab-search", 
    #                     {"query": query, "limit": limit}
    #                 )
                    
    #                 # Format the results
    #                 models = []
    #                 for model in result.get("models", []):
    #                     models.append({
    #                         "uid": model.get("uid", ""),
    #                         "name": model.get("name", "Unknown"),
    #                         "preview_link": f"https://sketchfab.com/models/{model.get('uid', '')}",
    #                         "thumbnail_url": model.get("thumbnails", {}).get("images", [{}])[0].get("url", ""),
    #                         "author": model.get("user", {}).get("username", "Unknown"),
    #                         "is_downloadable": model.get("isDownloadable", False)
    #                     })
    #                 return models
    #     except Exception as e:
    #         return [{"error": str(e)}]
    
    # async def get_model_details(self, uid: str) -> Dict[str, Any]:
    #     """Get detailed information about a specific model"""
    #     try:
    #         server_params = StdioServerParameters(
    #             command="node",
    #             args=["C:/Users/pa662/Documents/GitHub/sketchfab-mcp-server/build/index.js", "--api-key", self.api_key],
    #         )
            
    #         async with stdio_client(server_params) as (read, write):
    #             async with ClientSession(read, write) as session:
    #                 await session.initialize()
    #                 result = await session.tool_call(
    #                     "sketchfab-model-details", 
    #                     {"uid": uid}
    #                 )
    #                 return result.get("model", {})
    #     except Exception as e:
    #         return {"error": str(e)}