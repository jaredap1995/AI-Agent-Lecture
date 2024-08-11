import logging
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
import os
from dotenv import load_dotenv

load_dotenv()

#set up the prompt extractor template
PROMPT_EXTRACTOR_TEMPLATE = """Given an input string, if there is relevant user information extract it following the provided format instructions. \
If a city location is provided, attempt to provide the corresppnding zip code, do not make up answers. If you do not know the zip code, return an empty string.

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
name_schema = ResponseSchema(name="user_name",
                               description="The actual name of the user, may only be a first name",
                               data_type="string")
dob_schema = ResponseSchema(name="date of birth",
                                 description="The date of birth of the user in the format YYYY-MM-DD",
                                 data_type="string")
age_schema = ResponseSchema(name="age",
                                    description="The age of the user as an integer",
                                    data_type="integer")
color_schema = ResponseSchema(name="Favorite color",
                                 description="The favorite color of the user",
                                 data_type="string")
zip_code_schema = ResponseSchema(name="zip_code",
                                 description="The zip code of the city",
                                 data_type="integer")

response_schemas = [name_schema, dob_schema, age_schema, color_schema, zip_code_schema]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
llm = OpenAI(api_key=os.getenv('OPENAI_KEY'), temperature=0.7)

# Initialize the OpenAI LLM
prompt = PromptTemplate(
    input_variables=['input'], 
    template=PROMPT_EXTRACTOR_TEMPLATE,
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
    output_parser=output_parser
)
chain = LLMChain(llm=llm, prompt=prompt)

def main():
    questions = [
        "My name is John Doe, I am from San Francisco, I'm 30 years old, my favorite color is green",
        "What is the capital of France?",   
        "Hello, I’m Alice, 25 years old, my favorite color is blue, I am from New York",
    ]
    # questions = [input("Enter a string: ")]
    for question in questions:
        response = chain.run(question)
        logger.info(f"Question: {question}")
        logger.info(f"Response: {response}")

if __name__ == "__main__":
    main()

