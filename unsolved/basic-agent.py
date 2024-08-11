# Load environment variables from a .env file
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the OpenAI LLM
llm = OpenAI(api_key=os.getenv('OPENAI_KEY'), temperature=0)

# Define a simple prompt template
template = """
You are a personal assiatnt but you are only allowed to answer questions around programming. Do not answer any questions about any other topics.

If the question is not related to programming. Answer with "I am sorry, I can only answer questions about programming."
Question: {question}
"""

# Initialize the chain with the LLM and prompt
prompt = PromptTemplate(template=template, input_variables=["question"])

# Define a basic agent with the chain as the tool
chain = LLMChain(llm=llm, prompt=prompt)

# Initialize the agent
# chain.run("What time is it?")

def main():
    response = chain.invoke("What is metabolism?")
    print('\n\n ### RESPONSE ### \n\n', response['text'])

if __name__ == "__main__":
    main()
