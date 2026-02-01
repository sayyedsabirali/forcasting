import numpy as np
import pandas as pd

class FeatureEngineer:
    
    def transform(self, df: pd.DataFrame):
        df = df.copy()
        
        print(f"🔧 Starting feature engineering...")
        print(f"   Original columns: {df.columns.tolist()}")
        
        # Check and convert types
        if 'hour' in df.columns:
            # Make sure hour is float for operations
            if df['hour'].dtype != 'float64':
                df['hour'] = df['hour'].astype(float)
        
        if 'day_of_week' in df.columns:
            if df['day_of_week'].dtype != 'float64':
                df['day_of_week'] = df['day_of_week'].astype(float)
        
        # 1. Add peak hour feature
        if 'hour' in df.columns:
            peak_hours = [7, 8, 9, 10, 11, 17, 18, 19, 21]
            df['peak_hour'] = df['hour'].isin(peak_hours).astype(float)  # Keep as float
            
            print(f"   Added: peak_hour")
        
        # 2. Add weekend peak
        if 'day_of_week' in df.columns:
            # First create weekend boolean, then convert to float
            is_weekend = (df['day_of_week'] >= 5)
            df['weekend_peak'] = (
                is_weekend.astype(float) * 
                df['hour'].isin([9, 10, 11, 17, 18]).astype(float)
            ).astype(float)
            
            print(f"   Added: weekend_peak")
        
        # 3. Add high_demand_hour - FIXED VERSION
        if 'peak_hour' in df.columns and 'day_of_week' in df.columns:
            # Convert conditions to float properly
            is_weekend = (df['day_of_week'] >= 5).astype(float)
            df['high_demand_hour'] = (df['peak_hour'] * is_weekend).astype(float)
            
            print(f"   Added: high_demand_hour")
        
        # 4. Add interaction features
        if 'weather_temp' in df.columns:
            df['temp_peak_interaction'] = df['weather_temp'] * df.get('peak_hour', 0)
            print(f"   Added: temp_peak_interaction")
        
        # 5. Add recent high demand indicator
        if 'lag_1h' in df.columns:
            df['recent_high_demand'] = (df['lag_1h'] >= 2).astype(float)
            print(f"   Added: recent_high_demand")
        
        # Fill any NA
        df = df.fillna(0)
        
        print(f"✅ Feature engineering complete: {df.shape[1]} features")
        print(f"   New columns added: {[col for col in df.columns if col not in ['num_sessions', 'timestamp']]}")
        
        return df