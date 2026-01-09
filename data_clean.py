import pandas as pd
import numpy as np
import logging
from config import ENGINE
from sqlalchemy import inspect
import time

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

class Cleaner:
    
    def clean(self,df):
        
        if df is None:
            
            logging.error("Dataframe is Empty........")
        try:
            null_val_before=df.isnull().sum()/len(df)*100
            dup_val_before=df.duplicated().sum()/len(df)*100
            
            logging.info(f"Duplicates before: {dup_val_before}")
            logging.info(f"Nulls before: {null_val_before}")
            
            df_clean = df.dropna().drop_duplicates().reset_index(drop=True)
            
            null_val_after=df_clean.isnull().sum()/len(df)*100
            dup_val_after=df_clean.duplicated().sum()/len(df)*100
            
            logging.info(f"Duplicates After: {dup_val_after}")
            logging.info(f"Nulls After: {null_val_after}")
            
            inspector=inspect(ENGINE)
            
            TABLE_NAME="clean_data_info"
            if not inspector.has_table(TABLE_NAME):
                df_clean.to_sql(
                    name=TABLE_NAME,
                    con=ENGINE,
                    if_exists="fail",
                    index=False
                )
                logging.info(f"Clean_Data stored successfully in table '{TABLE_NAME}' at {time.asctime()}")
                    
            else:
                    
                logging.info("Table aready exist,Skipping load......")
                
            return df_clean
                
        
        
        except Exception as e:
            
            logging.error(f"Something Went Wrong....{e}")
            