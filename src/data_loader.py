# import pandas as pd
# from config import DATA_PATH
# import numpy as np

# class DataLoader:
#     def load_data(self):
#         df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
#         df = df.sort_values("timestamp").set_index("timestamp")

#         # Convert boolean columns to int
#         bool_cols = df.select_dtypes(include=['bool']).columns
#         if len(bool_cols) > 0:
#             df[bool_cols] = df[bool_cols].astype(int)

#         # Handle weather column (categorical → numeric)
#         if "weather_condition" in df.columns:
#             print("Encoding weather_condition column...")
#             df["weather_condition"] = df["weather_condition"].astype("category").cat.codes

#         # Now drop OTHER non-numeric columns (but not weather)
#         non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
#         if len(non_numeric_cols) > 0:
#             print(f"Dropping remaining non-numeric columns: {list(non_numeric_cols)}")
#             df = df.drop(columns=non_numeric_cols)

#         print(f"Final shape: {df.shape}")
#         return df
import pandas as pd
import numpy as np
from config import DATA_PATH

class DataLoader:
    def load_data(self):
        df = pd.read_csv(DATA_PATH)

        # Fix timestamp format
        df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True, errors="coerce")

        # Sort + index
        df = df.sort_values("timestamp").set_index("timestamp")

        # Encode station_id
        df["station_id"] = df["station_id"].astype("category").cat.codes

        # Encode weather condition
        df["weather_condition"] = df["weather_condition"].astype("category").cat.codes

        # Convert bool-like cols
        bool_cols = ["is_holiday", "is_idle_time"]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)

        # Fill numeric missing safely
        df = df.ffill().bfill()

        print(f"Final shape: {df.shape}")
        return df
