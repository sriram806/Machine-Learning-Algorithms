from src.utils import load_model
from src.train import PROJECT_PATH
from pathlib import Path
import pandas as pd

PROJECT_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_PATH/"model.pkl"

model = load_model(MODEL_PATH)

def validate_area(area:float):
    if area <= 0:
        raise ValueError(
            print(f"Area must greater than 0")
        )

def predict_house_price(area: float):
    sample = pd.DataFrame({
        "Area": [area]
    })

    predection = model.predict(sample)

    return predection[0]

def main():
    try:
        area = float(input("Enter The area of House: "))
        validate_area(area)
        prediction = predict_house_price(area)
        print(
            f"\nPredicted House Price : ₹{prediction:,.2f}"
        )

    except ValueError as error:
        print(error)

    except Exception as error:
        print(error)

if __name__ == "__main__":
    main()