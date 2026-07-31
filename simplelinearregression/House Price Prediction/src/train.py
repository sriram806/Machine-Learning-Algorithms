from pathlib import Path

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from utils import load_dataset, save_model

PROJECT_PATH = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_PATH / "data" / "house_price.csv"
MODEL_PATH = PROJECT_PATH / "model.pkl"
OUTPUT_DIR = PROJECT_PATH / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def print_header(title: str) -> None:
    """Print a formatted section header."""

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main() -> None:
    """Train and evaluate the Linear Regression model."""

    try:

        print_header("Loading Dataset")

        df = load_dataset(DATASET_PATH)

        print(f"Dataset Shape : {df.shape}")

        X = df[["Area"]]

        y = df["Price"]

        assert len(X) == len(
            y
        ), "Feature and Target row count mismatch."

        assert (
            X.shape[1] == 1
        ), "Simple Linear Regression requires one feature."

        print("Feature and Target validation passed.")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )

        print_header("Train-Test Split")

        print(f"Total Samples    : {len(df)}")
        print(f"Training Samples : {len(X_train)}")
        print(f"Testing Samples  : {len(X_test)}")

        print(f"X_train Shape    : {X_train.shape}")
        print(f"X_test Shape     : {X_test.shape}")

        print(f"y_train Shape    : {y_train.shape}")
        print(f"y_test Shape     : {y_test.shape}")

        model = LinearRegression()
        model.fit(X_train, y_train)

        print_header("Model Training")

        print("Model trained successfully.")

        print(f"Slope      : {model.coef_[0]:,.2f}")

        print(f"Intercept  : {model.intercept_:,.2f}")

        print("\nRegression Equation")

        print(
            f"Price = ({model.coef_[0]:,.2f} × Area) + ({model.intercept_:,.2f})"
        )

        assert hasattr(
            model,
            "coef_",
        ), "Model training failed."

        assert (
            model.coef_[0] > 0
        ), "Slope should be positive."

        y_pred = model.predict(X_test)

        results = pd.DataFrame(
            {
                "Area": X_test["Area"].values,
                "Actual Price": y_test.values,
                "Predicted Price": y_pred.round(2),
            }
        )

        results = (
            results.sort_values("Area")
            .reset_index(drop=True)
        )

        results["Residual"] = (
            results["Actual Price"]
            - results["Predicted Price"]
        )

        print_header("Test Set Predictions")

        print(results)

        new_house = pd.DataFrame({"Area": [2500]})

        predicted_price = model.predict(new_house)[0]

        print_header("New Prediction")

        print(f"Predicted Price for 2500 sq.ft. : Rs.{predicted_price:,.2f}")

        prediction_path = (OUTPUT_DIR/ "actual_vs_predicted.csv")

        results.to_csv(
            prediction_path,
            index=False,
        )

        print(f"Prediction results saved to:\n{prediction_path}")

        mae = mean_absolute_error(y_test,y_pred)
        mse = mean_squared_error(y_test,y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(y_test,y_pred)

        metrics = pd.DataFrame(
            {
                "Metric": [
                    "MAE",
                    "MSE",
                    "RMSE",
                    "R² Score",
                ],
                "Value": [
                    mae,
                    mse,
                    rmse,
                    r2,
                ],
            }
        )

        metrics_path = (OUTPUT_DIR / "metrics.csv")

        metrics.to_csv(
            metrics_path,
            index=False,
        )

        print_header("Evaluation Metrics")

        for metric, value in zip(metrics["Metric"],metrics["Value"]):print(f"{metric:<10}: {value:,.4f}")

        save_model(model,MODEL_PATH)

        print_header("Model Saved")

        print(f"Model saved successfully:\n{MODEL_PATH}")

        print("\nTraining Pipeline Completed Successfully.")

    except Exception as error:

        print_header("ERROR")

        print(error)

if __name__ == "__main__":
    main()