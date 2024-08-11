from langchain.agents import initialize_agent, Tool, AgentType
from langchain import OpenAI, LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import logging

load_dotenv()

# Example of a custom tool used for making jokes
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    
def dad_joke_tool(input_text: str) -> str:
    """Takes an input string and returns a dad joke related to it."""
    joke = "Why don't shoes ever get tired? Because they always get tied up!"
    logger.info("Using DadJokeTool")
    return joke


# Initialize the OpenAI LLM
llm = OpenAI(api_key=os.getenv('OPENAI_KEY'), temperature=0)

# Define the prompt
template = """You are an assistant with access to the following tools:
{tools}. Answer the question using the appropriate tool. If it is an exponent computation,
use the ArithmeticTool. If it is a joke, use the DadJokeTool.

Question: {question}
"""

# Initialize the chain with the LLM and prompt
prompt = PromptTemplate(template=template, input_variables=["tools", "question"])
chain = LLMChain(llm=llm, prompt=prompt)
tools = [
    Tool(name="DadJokeTool", func=dad_joke_tool, description="Generates dad jokes."),
    Tool(name="ArithmeticTool", func=arithmetic_tool, description="Performs basic exponent computation."),
]

agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

# Initialize the agent

def main():
    questions = ["What is 5 + 3?", "What is 4 squared?", "I forgot to untie my shoes, tell me something funny about it."]
    for question in questions:
        logger.info(f"Question: {question}")
        answer = agent.run(question)
        if "Using DadJokeTool" in answer or "Using ArithmeticTool" in answer:
            logger.info(answer)

if __name__ == "__main__":
    main()
