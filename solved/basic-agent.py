from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from dotenv import load_dotenv
import os

load_dotenv()

"""
Activity 1: Create a basic agent via langchain that answers a simple question.
"""

# # Initialize the OpenAI LLM
llm = OpenAI(api_key=os.getenv('OPENAI_KEY'), temperature=0.7)

# Define a simple prompt template
template = """
You are a personal trainer. Your job is to help people with their fitness goals and tell them HOW TO GET JACKED:
Question: {question}
"""
prompt = PromptTemplate(template=template, input_variables=["question"])

# Initialize the chain with the LLM and prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Define a basic agent with the chain as the tool
tools = [Tool(name="SimpleQA", func=chain, description="Simple question answering tool")]

# Initialize the agent
agent = initialize_agent(tools, llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

def main():
    question = "HOW DO I GET BIG???"
    print(agent.run(question))

if __name__ == "__main__":
    main()

