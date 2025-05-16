from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Type
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
import os
from dotenv import load_dotenv
import json
from enum import Enum
from langchain_core.tools import BaseTool
import time


load_dotenv()

router = APIRouter(prefix="/api/learning-style", tags=["learning-style"])

# Learning style types
class LearningStyle(str, Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    KINESTHETIC = "kinesthetic"
    READING_WRITING = "reading_writing"

# Pydantic models
class QuestionOption(BaseModel):
    id: str  # a, b, c, or d
    text: str
    learning_style: LearningStyle

class Question(BaseModel):
    id: int
    text: str
    options: List[QuestionOption]

class QuizResponse(BaseModel):
    questions: List[Question]
    quiz_id: str

# Initialize LLM
model = AzureChatOpenAI(
    azure_deployment="gpt-4o",
    api_version="2025-01-01-preview",
    temperature=0.2,
    max_tokens=None,
)

# Input schema for the learning style quiz generator
class LearningStyleQuizInput(BaseModel):
    num_questions: int = Field(
        default=5, 
        description="The number of learning style assessment questions to generate"
    )
    quiz_type: Optional[str] = Field(
        default=None,
        description="Optional specific type of quiz to generate (e.g., 'academic', 'professional', 'general')"
    )

class LearningStyleQuizTool(BaseTool):
    name: str = "learning_style_quiz_generator"
    description: str = """This tool generates learning style assessment questions to determine a person's preferred learning style.
    
    Each question will have 4 options that correspond to different learning styles:
    - Option A: Visual learning style (learns through seeing)
    - Option B: Auditory learning style (learns through hearing)
    - Option C: Kinesthetic learning style (learns through doing)
    - Option D: Reading/Writing learning style (learns through reading/writing)
    
    The questions are designed to be situation-based rather than direct preference questions.
    
    Use this tool when:
    1. You need to create a learning style assessment quiz
    2. You want to help someone understand their learning preferences
    3. You need to provide educational guidance based on learning styles
    
    The tool will return a list of questions, each with 4 options corresponding to different learning styles.
    """
    
    args_schema: Type[BaseModel] = LearningStyleQuizInput
    
    def _run(self, num_questions: int = 5, quiz_type: Optional[str] = None):
        pre_time = time.time()
        post_time = time.time()
        print(f"Time taken: {post_time - pre_time} seconds")



@router.get("/generate-quiz")
async def generate_learning_style_quiz():
    """Generate a learning style assessment quiz with 5 questions"""
    tools = [LearningStyleQuizTool()]

    system_prompt = """You are an expert educational psychologist specializing in learning styles.
    
    Create a set of learning style assessment questions that help determine a person's learning style.
    
    Each question must have 4 options (a, b, c, d) where:
    - Option A corresponds to VISUAL learning style
    - Option B corresponds to AUDITORY learning style
    - Option C corresponds to KINESTHETIC learning style
    - Option D corresponds to READING/WRITING learning style
    
    Make the questions situation-based rather than directly asking about preferences.
    Each option should reflect how a person with that learning style would naturally respond.
    
    The questions should cover different aspects of learning like:
    - How people prefer to receive new information
    - How they process and remember information
    - How they explain things to others
    - How they approach problem-solving
    - How they spend their free time
    
    RETURN YOUR RESPONSE AS A JSON ARRAY with this exact format:
    [
        {
        "question_id": "1",
        "question_text": "When trying to learn a new skill, you prefer to:",
        "option_a": { "text": "Watch a demonstration video", "learning_style": "visual" },
        "option_b": { "text": "Listen to someone explain the steps", "learning_style": "auditory" },
        "option_c": { "text": "Try it hands-on right away", "learning_style": "kinesthetic" },
        "option_d": { "text": "Read a detailed guide or manual", "learning_style": "reading_writing" }
        },
        // More questions following the same format
    ]
    
    IMPORTANT: The response MUST be a valid JSON array. Do not include any explanatory text before or after the JSON.
    """

    agent = create_react_agent(model=model, tools=tools, prompt=system_prompt)
    res = await agent.ainvoke({"messages": [("human", "Create a quiz with 5 questions.")]})
    print(res)
    response = res["messages"][-1].content
    return {"response": response}

