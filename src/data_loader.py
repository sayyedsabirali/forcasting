import pandas as pd
from config import DATA_PATH
import numpy as np

class DataLoader:
    def load_data(self):
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
        df = df.sort_values("timestamp").set_index("timestamp")
        
        print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Convert boolean columns to integer
        bool_cols = df.select_dtypes(include=['bool']).columns
        if len(bool_cols) > 0:
            df[bool_cols] = df[bool_cols].astype(int)
            print(f"   🔧 Converted boolean columns to int: {list(bool_cols)}")
        
        # Convert any other non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            print(f"   ⚠️ Dropping non-numeric columns: {list(non_numeric_cols)}")
            df = df.drop(columns=non_numeric_cols)
        
        print(f"   ✅ Final shape: {df.shape}")
        
        return df