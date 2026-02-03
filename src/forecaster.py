import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from config import MODEL_SAVE_PATH, TARGET_COL, FORECAST_HORIZON


class EVChargingForecaster:
    def __init__(self, model_name):
        self.model = joblib.load(f"{MODEL_SAVE_PATH}/{model_name}.pkl")
        self.feature_columns = joblib.load(f"{MODEL_SAVE_PATH}/feature_columns.pkl")
        print(f"✅ Loaded model: {model_name}")


    def forecast(self, df):

        history = df.copy()
        history[TARGET_COL] = history[TARGET_COL].astype(int)

        future_hours = df.index[-FORECAST_HORIZON:]
        history = df.iloc[:-FORECAST_HORIZON].copy()

        preds = []


        for ts in future_hours:
            actual_row = df.loc[ts]

            row = {
                "hour": ts.hour,
                "dayofweek": ts.dayofweek,
                "is_weekend": int(ts.dayofweek >= 5),
                "station_id": actual_row["station_id"],
                "weather_temp": actual_row["weather_temp"],
                "weather_condition": actual_row["weather_condition"],
                "price_inr": actual_row["price_inr"],
                "is_holiday": actual_row["is_holiday"],
            }
            

            X_pred = pd.DataFrame([row])
            X_pred = X_pred.reindex(columns=self.feature_columns, fill_value=0)
            print("X_pred sample:")
            print(X_pred)
            print("ROW KEYS:", row.keys())
            exit()


            y_pred = self.model.predict(X_pred)[0]
            preds.append(y_pred)

            new_row = pd.DataFrame({TARGET_COL: [y_pred]}, index=[ts])
            history = pd.concat([history, new_row])

        forecast_df = pd.DataFrame({
            "timestamp": future_hours,
            "predicted_sessions": preds
        })

        return forecast_df
