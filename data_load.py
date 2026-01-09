import pandas as pd
import os
import logging
from config import ENGINE,TABLE_NAME
from sqlalchemy import inspect
import time

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

class Load_File:
    
    def __init__(self,engine=ENGINE):
        self.engine=engine
    
    def load(self,file_path):
        
        if os.path.exists(file_path):
            
            try:
                
                df=pd.read_csv(file_path)
                
                df.rename(columns={"cb_person_default_on_file":"person_default"},inplace=True)
                df = df.where(pd.notnull(df), None)  #covert nan to None
                
                logging.info(f"\n{df.head(5).to_string()}")
                inspector=inspect(self.engine)
                
                if not inspector.has_table(table_name=TABLE_NAME):
                  
                    df.to_sql(
                    name=TABLE_NAME,
                    con=self.engine,
                    if_exists="fail",
                    index=False   
                    )
                    
                    logging.info(f"Data stored successfully in table '{TABLE_NAME}' at {time.asctime()}")
                    
                else:
                    
                    logging.info("Table aready exist,Skipping load......")
                
                return df
                
            except Exception as e:
                logging.error(f"Something went wrong {e}")
                

    