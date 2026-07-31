from pathlib import Path
import pandas as pd
import joblib

PROJECT_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_PATH / "model.pkl"
DATASET_PATH = PROJECT_PATH / "data" / "house_price.csv"

def load_dataset(file: Path) -> pd.DataFrame:
    try:
        df= pd.read_csv(file)
        print("Dataset Loaded Successfully")
        return df

    except FileNotFoundError:
        raise FileNotFoundError 
        (
            f"Dataset not found: {file}"
        )
    
    except pd.errors.EmptyDataError:
        raise ValueError(
            f"Dataset is Empty"
        )

    except Exception as error:
        raise RuntimeError(
            f"Unexpected Error: {error}"
        )

    return df


def save_model(model, file: Path):
    joblib.dump(model, file)
    if MODEL_PATH.exists():
        print("Model saved successfully.")
    else:
        print("Model save failed.")
    
    print(f"Model saved to:{MODEL_PATH}")

def load_model(file: Path):
    model = joblib.load(file)
    print("Model Loaded Successsfully")
    return model
