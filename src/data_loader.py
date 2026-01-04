import os
import pandas as pd
from ucimlrepo import fetch_ucirepo
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def load_data(save_path="data/raw/heart_disease.csv"):
    """
    Downloads the Heart Disease UCI dataset and saves it to CSV.
    """
    try:
        logging.info("Attempting to download Heart Disease UCI dataset...")
        
        # fetch dataset 
        heart_disease = fetch_ucirepo(id=45) 
        
        if not heart_disease.data:
            raise ValueError("Downloaded data is empty.")

        # data (as pandas dataframes) 
        X = heart_disease.data.features 
        y = heart_disease.data.targets 
        
        # Combine features and target
        df = pd.concat([X, y], axis=1)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(save_path, index=False)
        logging.info(f"Dataset saved successfully to {save_path}")
        return df
        
    except Exception as e:
        logging.error(f"Failed to download data: {e}")
        raise e

if __name__ == "__main__":
    load_data()
