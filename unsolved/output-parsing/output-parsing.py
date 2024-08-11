import logging
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
import os
from dotenv import load_dotenv

load_dotenv()

#set up the prompt extractor template



# Set up logging

# Define your response schemas

# Initialize the OpenAI LLM

