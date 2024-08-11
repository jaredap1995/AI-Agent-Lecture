from langchain.agents import initialize_agent, Tool, AgentType
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import logging

load_dotenv()

"""
FWIW: I do not love these examples of tools and they don't run very well so don't really try and and use them, they are more just examples of what a 
tool could do in a hopefully simple way. 

Hopefully the next examples demonstrate their utility better.

"""

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Example of a custom tool used for making jokes
def dad_joke_tool(input_text: str) -> str:
    """Takes an input string and returns a dad joke related to it."""
    joke = "Why don't shoes ever get tired? Because they always get tied up!"
    logger.info("Using DadJokeTool")
    return joke

# Tool for basic arithmetic
def arithmetic_tool(input_text: str) -> str:
    """Performs basic exponent computations."""
    try:
        num = int(input_text)
        result = num ** 2
        logger.info("Using ArithmeticTool")
        return f"The square of {num} is {result}."
    except ValueError:
        return "Error: input must be an integer."

# Create tools
tools = [
    Tool(name="DadJokeTool", func=dad_joke_tool, description="Generates dad jokes."),
    Tool(name="ArithmeticTool", func=arithmetic_tool, description="Performs basic exponent computation."),
]

# Initialize the OpenAI LLM
llm = OpenAI(api_key=os.getenv('OPENAI_KEY'), temperature=0.7)

# Define the prompt
template = """You are an assistant with access to the following tools: 
{tools}. Answer the question using the appropriate tool. If it is an exponent computation,
use the ArithmeticTool. If it is a joke, use the DadJokeTool.
"""
prompt = PromptTemplate(template=template, input_variables=["tools", "question"])

# Initialize the chain with the LLM and prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Initialize the agent
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

def main():
    questions = [
        "What is 5 + 3?", 
        "What is 4 squared?", 
        "I forgot to untie my shoes, tell me something funny about it."
    ]
    
    for question in questions:
        logger.info(f"Question: {question}")
        answer = agent.run(question)
        
        # Log which tool was used or if no tool was used
        if "Using DadJokeTool" in answer or "Using ArithmeticTool" in answer:
            pass  # Already logged within the tool function
        else:
            logger.info("No specific tool was used.")
        
        print(f"\n ## Question: {question}")
        print(f'\n\n ### {answer}\n')

if __name__ == "__main__":
    main()


