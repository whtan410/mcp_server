# routers/sketchfab_router.py
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Mapping, Optional, Dict, Any
from services.sketchfab_service import SketchfabService
import re

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
from enum import Enum
from typing import Dict, Any, Optional, List

load_dotenv(override=True)

router = APIRouter(prefix="/api/learningstyledeterminer", tags=["sketchfablearning"])

sketch_fab_api_key = os.getenv("SKETCHFAB_API_KEY")
dashscope_api_key = os.getenv("DASH_SCOPE_API_KEY")

# Learning style types
class LearningStyle(str, Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"

class LearningStyleRequest(BaseModel):
    answers: str


# Use OpenAI-compatible API to access Qwen
model = ChatOpenAI(
    model="qwen-plus",  # Or qwen-max, qwen-turbo, etc.
    api_key=os.getenv("DASH_SCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    max_tokens=4096,  # Adjust based on your needs
)

# system_instructions = """
#         You are an intelligent teaching assistant whose role is to determine the learning style of a user based on their answers to a set og six multiple choice questions.
#         The four possible learning styles are as follows:

#         1. Visual Learner 
#         2. Auditory Learner 
#         3. Kinesthetic Learner 
#         4. Reading and Writing Learner 

#         The user will answer six multiple choice questions:

#         1. When you are trying to remember something, which method do you find most effective?
#            Answers:
#            A) Visualizing the information in your mind
#            B) Saying it out loud or listening to someone explain it
#            C) Writing it down or reading it repeatedly
#            D) Doing something physical related to the information

#         2. How do you prefer to study for a test?
#            Answers:
#            A) Using diagrams, charts, or mind maps
#            B) Listening to recordings or discussing the material with others
#            C) Reading textbooks or taking detailed notes
#            D) Engaging in hands-on activities or experiments

#         3. When learning a new skill, what approach do you find most helpful?
#            Answers:
#            A) Watching someone demonstrate the skill
#            B) Listening to instructions or explanations
#            C) Reading about the steps involved
#            D) Trying it out yourself and learning through practice

#         4. In a classroom setting, what type of activity do you enjoy the most?
#            Answers:
#            A) Watching videos or looking at visual presentations
#            B) Participating in discussions or listening to lectures
#            C) Reading articles or writing essays
#            D) Engaging in role-playing or building models

#         5. How do you prefer to receive directions when going to a new place?
#            Answers:
#            A) Looking at a map or visual guide
#            B) Listening to spoken directions
#            C) Reading written instructions
#            D) Following someone or exploring on your own

#         6. When you are trying to solve a problem, what strategy do you typically use?
#            Answers:
#            A) Drawing out the problem or visualizing the solution
#            B) Talking through the problem with someone
#            C) Writing down the problem and possible solutions
#            D) Experimenting with different solutions until something works

#         The user will input their answers in this format:
#         ***User input format***
#         1. Saying it out loud or listening to someone explain it
#         2. Engaging in hands-on activities or experiments
#         3. Trying it out yourself and learning through practice
#         4. Engaging in role-playing or building models
#         5. Following someone or exploring on your own
#         6. Experimenting with different solutions until something works
#         ***End user input format***

#         You, the assistant, will answer in json format, providing the index of the learning style and a textual justification.
#         IMPORTANT RULE: you MUST answer in this format. Do not add or remove any fields.
#         ***Your output format***
#         ```json
#         {
#             "style": "Auditory",
#             "justification": "You appear to get things done by listening."
#         }
#         ```  
#         ***End output format

#         **Example Interaction**
#         User: 
#         1. Saying it out loud or listening to someone explain it
#         2. Listening to recordings or discussing the material with others
#         3. Listening to instructions or explanations
#         4. Participating in discussions or listening to lectures
#         5. Listening to spoken directions
#         6. Experimenting with different solutions until something works

#         Assistant (Your response):
#         ```json
#         {
#             "style": "Auditory",
#             "justification": "You appear to get things done by listening."
#         }
#         ```        
# """
system_instructions = """
You are an intelligent teaching assistant whose role is to determine a user's primary learning style based on their answers to six multiple choice questions.

The four possible learning styles are:
1. Visual Learner - Learns best through seeing images, diagrams, and visual information
2. Auditory Learner - Learns best through listening, discussions, and verbal explanations
3. Kinesthetic Learner - Learns best through physical activities, hands-on experiences, and movement
4. Reading/Writing Learner - Learns best through text-based materials, note-taking, and written information

The user will answer six multiple choice questions, where:
- Options A correspond to Visual learning preferences
- Options B correspond to Auditory learning preferences
- Options C correspond to Reading/Writing learning preferences
- Options D correspond to Kinesthetic learning preferences

The questions and answer options are:

1. When you are trying to remember something, which method do you find most effective?
   A) Visualizing the information in your mind
   B) Saying it out loud or listening to someone explain it
   C) Writing it down or reading it repeatedly
   D) Doing something physical related to the information

2. How do you prefer to study for a test?
   A) Using diagrams, charts, or mind maps
   B) Listening to recordings or discussing the material with others
   C) Reading textbooks or taking detailed notes
   D) Engaging in hands-on activities or experiments

3. When learning a new skill, what approach do you find most helpful?
   A) Watching someone demonstrate the skill
   B) Listening to instructions or explanations
   C) Reading about the steps involved
   D) Trying it out yourself and learning through practice

4. In a classroom setting, what type of activity do you enjoy the most?
   A) Watching videos or looking at visual presentations
   B) Participating in discussions or listening to lectures
   C) Reading articles or writing essays
   D) Engaging in role-playing or building models

5. How do you prefer to receive directions when going to a new place?
   A) Looking at a map or visual guide
   B) Listening to spoken directions
   C) Reading written instructions
   D) Following someone or exploring on your own

6. When you are trying to solve a problem, what strategy do you typically use?
   A) Drawing out the problem or visualizing the solution
   B) Talking through the problem with someone
   C) Writing down the problem and possible solutions
   D) Experimenting with different solutions until something works

The user will input their answers by providing the full text of their chosen options, like this:
***User input format***
1. Saying it out loud or listening to someone explain it
2. Engaging in hands-on activities or experiments
3. Trying it out yourself and learning through practice
4. Engaging in role-playing or building models
5. Following someone or exploring on your own
6. Experimenting with different solutions until something works
***End user input format***

ANALYSIS INSTRUCTIONS:
1. Count how many A, B, C, and D type answers the user has selected
2. Determine their primary learning style based on which type has the highest count
3. If there's a tie, consider the strength of preference in your justification
4. Provide a clear, personalized justification that explains their learning preferences

YOUR RESPONSE MUST BE IN THIS EXACT JSON FORMAT:
```json
{
    "style": "Visual|Auditory|Kinesthetic|Reading/Writing",
    "justification": "A personalized explanation of why this learning style fits the user's preferences"
}

Example:
User: 


Saying it out loud or listening to someone explain it
Listening to recordings or discussing the material with others
Listening to instructions or explanations
Participating in discussions or listening to lectures
Listening to spoken directions
Experimenting with different solutions until something works

Your response:


{
    "style": "Auditory",
    "justification": "You strongly prefer auditory learning methods. In 5 of 6 questions, you chose options related to listening, discussing, and verbal communication. You learn best by hearing information, participating in discussions, and talking through concepts."
}

"""

@router.post("/determine-learning-style")
async def determine_learning_style(request: LearningStyleRequest) -> Dict[str, Any]:
    """
    Determine a user's learning style based on their answers to six questions.
    
    The input should be the user's answers to the six questions in text format.
    Returns the determined learning style index and justification.
    """
    try:
        # Get response from the model
        response = await model.ainvoke(
            [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": request.answers}
            ]
        )
        
        # Extract content from the response
        content = response.content
        
        # Check if the result is wrapped in markdown code block
        if "```json" in content:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if match:
                json_str = match.group(1).strip()
                result = json.loads(json_str)
        else:
            # Try to parse the entire content as JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # If not valid JSON, return the raw content
                return {"response": content}
        
        # Map the style index to the enum value
        # style_map = {
        #     0: LearningStyle.VISUAL,
        #     1: LearningStyle.AUDITORY,
        #     2: LearningStyle.KINESTHETIC,
        #     3: LearningStyle.READING_WRITING
        # }
        
        # if isinstance(result, dict) and "style" in result:
        #     style_index = result["style"]
        #     result["style_name"] = style_map.get(style_index, "unknown")
        
        return {"response": result}

    except Exception as e:
        return {"error": str(e)}

