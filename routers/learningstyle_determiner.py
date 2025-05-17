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

router = APIRouter(prefix="/api/learning-style-determiner", tags=["Learning Style Determiner"])

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

# class OneEnum(str, Enum):
#    VISUAL="Visualizing the information in your mind"
#    AUDITORY="Saying it out loud or listening to someone explain it"
#    KINESTHETIC="Writing it down or reading it repeatedly"
#    READING_WRITING="Doing something physical related to the information"


# class TwoEnum(str, Enum):
#     VISUAL = "Using diagrams, charts, or mind maps"
#     AUDITORY = "Listening to recordings or discussing the material with others"
#     READING_WRITING = "Reading textbooks or taking detailed notes"
#     KINESTHETIC = "Engaging in hands-on activities or experiments"

# class ThreeEnum(str, Enum):
#     VISUAL = "Watching someone demonstrate the skill"
#     AUDITORY = "Listening to instructions or explanations"
#     READING_WRITING = "Reading about the steps involved"
#     KINESTHETIC = "Trying it out yourself and learning through practice"

# class FourEnum(str, Enum):
#     VISUAL = "Watching videos or looking at visual presentations"
#     AUDITORY = "Participating in discussions or listening to lectures"
#     READING_WRITING = "Reading articles or writing essays"
#     KINESTHETIC = "Engaging in role-playing or building models"

# class FiveEnum(str, Enum):
#     VISUAL = "Looking at a map or visual guide"
#     AUDITORY = "Listening to spoken directions"
#     READING_WRITING = "Reading written instructions"
#     KINESTHETIC = "Following someone or exploring on your own"

# class SixEnum(str, Enum):
#     VISUAL = "Drawing out the problem or visualizing the solution"
#     AUDITORY = "Talking through the problem with someone"
#     READING_WRITING = "Writing down the problem and possible solutions"
#     KINESTHETIC = "Experimenting with different solutions until something works"

class LearningAnswers(BaseModel):
    one: str
    two:str
    three: str
    four: str
    five: str
    six: str


# Use OpenAI-compatible API to access Qwen
model = ChatOpenAI(
    model="qwen-max",  # Or qwen-max, qwen-turbo, etc.
    api_key=os.getenv("DASH_SCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    max_tokens=4096,  # Adjust based on your needs
)

system_instructions = """
You are an intelligent teaching assistant whose role is to determine a user's primary learning style based on their answers to six multiple choice questions.

The four possible learning styles are:
1. Visual Learner - Learns best through seeing images, diagrams, and visual information. This style has an index of 1.
2. Auditory Learner - Learns best through listening, discussions, and verbal explanations. This style has an index of 2.
3. Kinesthetic Learner - Learns best through physical activities, hands-on experiences, and movement. This style has an index of 3.
4. Reading/Writing Learner - Learns best through text-based materials, note-taking, and written information. This style has an index of 4.

The questions and answer options are:
1. Your teacher is showing the class how to do a science experiment. How do you understand it best?

A. By watching what the teacher is doing.

B. By listening carefully to what the teacher says.

C. By reading the steps on the worksheet.

D. By trying the experiment yourself.

2. You need to learn a poem for homework. What do you do?

A. Draw pictures to help remember the parts.

B. Say it out loud over and over again.

C. Write it down a few times to help you remember.

D. Act it out with hand movements or walk around while practicing.

3. When you're trying to figure out how a new game works, what helps you the most?

A. Looking at pictures or diagrams.

B. Having someone explain it to you.

C. Reading the instructions.

D. Jumping in and trying it out.

4. Your class is learning about volcanoes. What do you enjoy most?

A. Watching videos or looking at volcano pictures.

B. Listening to the teacher tell exciting facts.

C. Reading about volcanoes in a book or online.

D. Building a volcano model or doing a project.

5. You’re asked to give a short presentation in class. How do you prepare?

A. Make a colorful poster or slideshow.

B. Practice talking out loud with a friend.

C. Write down everything you want to say.

D. Rehearse by moving around and acting it out.

6. You’re learning a new song in music class. What helps you remember it?

A. Watching the teacher’s hand movements or sheet music.

B. Singing it over and over again.

C. Reading the lyrics quietly to yourself.

D. Clapping or tapping the beat while singing.

The user will input their answers by providing the full text of their chosen options, like this:
***User input format***
1. By watching what the teacher is doing.
2. Draw pictures to help remember the parts.
3. Looking at pictures or diagrams.
4. Watching videos or looking at volcano pictures.
5. Make a colorful poster or slideshow.
6. You’re learning a new song in music class. What helps you remember it?
***End user input format***

Provide a clear, personalized justification that explains their learning preferences

YOUR RESPONSE MUST BE IN THIS EXACT JSON FORMAT:
```json
{
    "style": "Visual|Auditory|Kinesthetic|Reading/Writing",
    "style_index": 1|2|3|4
    "justification": "A personalized explanation of why this learning style fits the user's preferences"
}

Example:
User: 


1. By watching what the teacher is doing.
2. Draw pictures to help remember the parts.
3. Looking at pictures or diagrams.
4. Watching videos or looking at volcano pictures.
5. Make a colorful poster or slideshow.
6. You’re learning a new song in music class. What helps you remember it?

Your response:


{
    "style": "Auditory",
    "style_index": "2",
    "justification": "You strongly prefer auditory learning methods. In 5 of 6 questions, you chose options related to listening, discussing, and verbal communication. You learn best by hearing information, participating in discussions, and talking through concepts."
}

"""

@router.post("/determine-learning-style")
async def determine_learning_style(request: LearningAnswers) -> Dict[str, Any]:
    """
    Determine a user's learning style based on their answers to six questions.
    
    The input should be the user's answers to the six questions in json format. It will then be parssed into a single string to query Qwen.
    Returns the determined learning style, style index and justification.

    styles and their indexes:\n
    Visual: 1\n
    Auditory: 2\n
    Kinesthetic: 3\n
    Reading/Writing: 4\n

    Example json input:\n
    {\n
        "one": "By trying the experiment yourself.",\n
        "two": "Draw pictures to help remember the parts.",\n
        "three": "Jumping in and trying it out.",\n
        "four": "Watching videos or looking at volcano pictures.",\n
        "five": "Make a colorful poster or slideshow.",\n
        "six": "Watching the teacher’s hand movements or sheet music."\n
    }\n

    Example json output:\n
    {\n
        "response": {\n
            "style": "Visual",\n
            "style_index": 1,\n
            "justification": "Your answers indicate a strong preference for visual learning. In most of your responses, you chose options that involve seeing images, diagrams, or visual aids to understand and remember information. For example, you prefer watching videos, looking at pictures, and observing hand movements or sheet music. These preferences show that you learn best when information is presented visually."\n
        }\n
    }\n

    """
    try:
        answers = f"1. {request.one}\n2. {request.two}\n3. {request.three}\n4. {request.four}\n5. {request.five}\n6. {request.six}"

        # Get response from the model
        response = await model.ainvoke(
            [
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": answers}
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

        # fallback checker in case Qwen messes up the index
        styles = ["visual", "auditory", "kinesthetic", "reading/writing"]
        for i in range(len(styles)):
            if 'style' in result and 'style_index' in result and result["style"].lower() == styles[i]:
                result["style_index"] = i+1
                break
        
        return {"response": result}

    except Exception as e:
        return {"error": str(e)}

