
from joblib import load,dump
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Deployment:
    
    def deploy(self, model):
        
        if model is None:
            raise ValueError("Model is None. Cannot deploy an empty model.")
        
        try:
            dump(model, "name.joblib")
            logging.info("Model saved successfully.")
        
        except Exception as e:
            logging.error(f"Model deployment failed: {e}")
            raise
