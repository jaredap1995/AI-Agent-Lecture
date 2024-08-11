import logging
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
import os
from dotenv import load_dotenv

load_dotenv()


PROMPT_EXTRACTOR_TEMPLATE = """Given an input string, if there is relevant user information extract it following the provided format instructions. \
If there is no relevant information, return an empty string. \

Format Instructions:
{format_instructions}

Original Text:
{input}
"""



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Define your response schemas
title_schema = ResponseSchema(name="user_name",
                               description="The actual name of the user, may only be a first name",
                               data_type="string")
author_schema = ResponseSchema(name="date of birth",
                               description="The date of birth of the user in the format YYYY-MM-DD",
                               data_type="string")
keywords_schema = ResponseSchema(name="age",
                                 description="The age of the user as an integer",
                                 data_type="integer")
publisher_schema = ResponseSchema(name="Favorite color",
                                  description="The favorite color of the user",
                                  data_type="string")

response_schemas = [title_schema, author_schema, keywords_schema, publisher_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

# Initialize the OpenAI LLM
llm = OpenAI(api_key=os.getenv('OPENAI_KEY'), temperature=0.7)
prompt = PromptTemplate(
    input_variables=['input'], 
    template=PROMPT_EXTRACTOR_TEMPLATE,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    output_parser=output_parser
)
chain = LLMChain(llm=llm, prompt=prompt)

def main():
    questions = [
        "My name is John Doe, I'm 30 years old, my email is johndoe@example.com, and I enjoy running and yoga.",
        "What is the capital of France?",
        "Hello, I’m Alice, 25 years old, email alice@example.com. I like painting and cycling.",
    ]
    
    for question in questions:
        logger.info(f"Question: {question}")
        response = chain.run({'input': question})
        print(type(response))
        print(f"\n ## Answer: {response}\n")

if __name__ == "__main__":
    main()


"""
Do you notice the power of the output parser in the code above? It allows the declaration of structured outputs in a universal language
like JSON and can be combined with a tool to automatically extract information and potentially interact with a database.
For example, imagine chaining this together with an agent action that updates your database? You would never need to manually parse your data or update your database.

"""