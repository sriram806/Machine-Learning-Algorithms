import joblib
from pathlib import Path
import pandas as pd


PROJECT_PATH = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_PATH/"data"/"temperature_icecream_sales.csv"

def load_dataset(file:Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file)
        print("Dataset Loaded Successfully")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(
            f"File not found at {file}"
        )

    except pd.errors.EmptyDataError:
        raise ValueError(
            f"File is empty"
        )
    
    except Exception as error:
        raise RuntimeError(
            f"Something went worng: {error}"
        )

def model_save(model, file:Path):
    try:
        joblib.dump(model,file)
        print(f"Model saved successfully to {file}")
    except Exception as error:
        raise RuntimeError(
            f"Failed to save model: {error}"
        )

def load_model(file:Path) -> object:
    try:
        model = joblib.load(file)
        print(f"Model loaded successfully from {file}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model file not found at {file}"
        )
    except Exception as error:
        raise RuntimeError(
            f"Failed to load model: {error}"
        )