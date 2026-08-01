from utils import load_model
from pathlib import Path
import pandas as pd

PROJECT_PATH = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_PATH / "models" / "model.pkl"

model = load_model(MODEL_PATH)


def validate_temperature(temperature: float):
    if temperature <= 0:
        raise ValueError(
            f"Temperature must be greater than 0, got {temperature}"
        )


def predict_ice_cream_sales(temperature: float) -> float:
    validate_temperature(temperature)
    sample = pd.DataFrame({
        "Temperature": [temperature]
    })
    prediction = model.predict(sample)
    return round(prediction[0], 2)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Ice Cream Sales Prediction".center(60))
    print("=" * 60)

    test_temps = [15.0, 25.0, 35.0, 45.0]
    for temp in test_temps:
        sales = predict_ice_cream_sales(temp)
        print(f"  Temperature: {temp}C  ->  Predicted Sales: {sales} units")

    print("=" * 60)
