
import pandas as pd
import json
import os
from langchain import OpenAI
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from langchain.agents import create_sql_agent, AgentType
from dotenv import load_dotenv

load_dotenv()

class DataQueryExecutor:
    def __init__(self, config_file='config.json'):
            # with open(config_file, 'r') as file:
            #     self.config = json.load(file)
            
            # declare for flexibility, others remain unchanged

            self.database_url = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@localhost:5432/{os.getenv('DB_NAME')}"
            self.db = SQLDatabase.from_uri(self.database_url)
            self.llm = ChatOpenAI(temperature=0.0, model="gpt-3.5-turbo", openai_api_key=os.getenv('OPENAI_KEY'), max_tokens=1024)
            self.toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)

    def run_sql_agent(self, prompt):
        agent_executor = create_sql_agent(
            llm=self.llm,
            toolkit=self.toolkit,
            verbose=True,
            handle_parsing_errors=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        response = agent_executor.run(prompt)
        return response
    

    def save_to_csv(self, df, filename):
        export_path = os.path.join(os.getcwd(), 'exports')
        
        if not os.path.exists(export_path):
            os.makedirs(export_path)

        file_path = os.path.join(export_path, filename)
        df.to_csv(file_path, index=False)

    def main(self):
        router = input("Would data would you like to access?")
        # We can add the router chain here contingent on the data
        # query = self.routerChain(router)
        response = self.run_sql_agent(router) # replace with query

if __name__ == "__main__":
    executor = DataQueryExecutor()
    while True:
        executor.main()
        rerun = input("Would you like to run another query? (y/n): ")
        if rerun == 'n':
            break

