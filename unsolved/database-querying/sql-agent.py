
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
    def __init__(self):
        pass

    def run_sql_agent(self, prompt):
        pass
    

    def save_to_csv(self, df, filename):
        pass

    def main(self):
        pass

if __name__ == "__main__":
    pass

