from utils import model_save
from utils import DATASET_PATH,PROJECT_PATH,load_dataset
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

OUTPUT_PATH = PROJECT_PATH/"output"
MODEL_PATH = PROJECT_PATH/"models"/"model.pkl"

def print_header(title:str) -> None:
    print("\n"+"="*80)
    print(title.center(80))
    print("="*80)

def main() -> None:
    print_header("Training Simple Linear Regression Model")

    try:
        df = load_dataset(DATASET_PATH)
        print("Dataset loaded successfully")
    except Exception as error:
        raise ValueError(
            f"Failed to load dataset: {error}"
        )

    X = df[["Temperature"]]
    y = df["IceCreamSales"]

    assert len(X) == len(y), "Feature and Target Count mismatched"
    

    assert X.shape[1] >= 1, "At least one feature is required to predict the Ice Cream Sales"

    print_header("Splitting Dataset into Trainuing and Testing")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    print_header("Model Trainig")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model Train Successfully")

    print_header("Evaluation")

    print(f"Slope: {model.coef_[0]}")
    print(f"Intercept: {model.intercept_}")

    assert model.coef_[0] > 0, "Slope should be positive as Temperature increases Ice cream Sales also increases"
    assert hasattr(model, "coef_"), "Slope not found"
    assert hasattr(model, "intercept_"), "Intercept not found"

    print(f"Regression Line:\n Icecream Sale = {model.coef_[0]}*Temperature + {model.intercept_}")

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test,y_pred)

    print(f"MAE: {mae}")
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"R2 Score: {r2}")

    results = pd.DataFrame({
        "Temperature": X_test["Temperature"],
        "Actual Sales": y_test,
        "Predicted Sales": y_pred
    })

    results.sort_values("Temperature",inplace=True)
    results["Residual"] = (
        results["Actual Sales"]
        - results["Predicted Sales"]
    )

    print(results)

    prediction_path = OUTPUT_PATH/"actual_vs_predicted.csv"
    results.to_csv(prediction_path,index=False)

    print(f"Results saved to: {prediction_path}")

    metrics = pd.DataFrame({
        "Metrics":[
            "MAE",
            "MSE",
            "RMSE",
            "R2 Score"
        ],
        "Values":[
            mae,
            mse,
            rmse,
            r2
        ]
    })

    metrics_path = OUTPUT_PATH/"metrics.csv"
    metrics.to_csv(metrics_path,index=False)

    print(f"Metrics saved to: {metrics_path}")

    model_save(model,MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()