# # import joblib
# # import pandas as pd
# # import numpy as np
# # from datetime import timedelta
# # from config import MODEL_SAVE_PATH, TARGET_COL, FORECAST_HORIZON

# # class EVChargingForecaster:
# #     def __init__(self, model_name="RandomForest"):
# #         """Constructor that accepts model_name parameter"""
# #         self.model_name = model_name
# #         try:
# #             self.model = joblib.load(f"{MODEL_SAVE_PATH}/{model_name}.pkl")
# #             print(f"   ✅ Loaded model: {model_name}")
# #         except Exception as e:
# #             print(f"⚠️ Model {model_name}.pkl not found: {e}")
# #             raise
    
# #     def forecast(self, df):
# #         print(f"🔮 Forecasting with {self.model_name}...")
        
# #         # Use last 24 hours
# #         if len(df) >= 24:
# #             base_df = df.iloc[-24:].copy()
# #         else:
# #             base_df = df.copy()
        
# #         # Convert target to float
# #         base_df = base_df.copy()
# #         base_df[TARGET_COL] = base_df[TARGET_COL].astype(float)
        
# #         forecast_df = base_df.copy()
# #         results = []
        
# #         for i in range(FORECAST_HORIZON):
# #             next_time = forecast_df.index[-1] + timedelta(hours=1)
            
# #             # Copy last row
# #             new_row = forecast_df.iloc[-1].copy()
            
# #             # Update time features
# #             new_row['hour'] = next_time.hour
# #             new_row['day_of_week'] = next_time.weekday()
# #             new_row['is_weekend'] = 1 if next_time.weekday() >= 5 else 0
            
# #             # Add to dataframe
# #             forecast_df.loc[next_time] = new_row
            
# #             # Predict
# #             X_pred = forecast_df.drop(columns=[TARGET_COL]).iloc[[-1]]
            
# #             try:
# #                 point_pred = float(self.model.predict(X_pred)[0])
# #                 point_pred = max(0.0, point_pred)
# #                 point_pred = min(point_pred, 3.5)
# #             except Exception as e:
# #                 print(f"   ⚠️ Prediction error: {e}")
# #                 point_pred = float(forecast_df[TARGET_COL].tail(3).mean())
            
# #             # Update dataframe
# #             forecast_df.loc[next_time, TARGET_COL] = float(point_pred)
            
# #             # Store result
# #             results.append({
# #                 "timestamp": str(next_time),
# #                 "predicted_sessions": round(point_pred, 2),
# #                 "confidence": 0.8,
# #                 "confidence_lower": max(0, round(point_pred * 0.7, 2)),
# #                 "confidence_upper": round(point_pred * 1.3, 2)
# #             })
        
# #         # Print stats
# #         pred_values = [r['predicted_sessions'] for r in results]
# #         print(f"   ✅ Forecast complete: {len(results)} hours")
# #         print(f"   📊 Min: {min(pred_values):.2f}, Max: {max(pred_values):.2f}, Mean: {np.mean(pred_values):.2f}")
        
# #         return results


# import joblib
# import pandas as pd
# import numpy as np
# from datetime import timedelta
# from config import MODEL_SAVE_PATH, TARGET_COL, FORECAST_HORIZON

# class EVChargingForecaster:
#     def __init__(self, model_name="RandomForest_Classification"):
#         self.model_name = model_name
#         try:
#             self.model = joblib.load(f"{MODEL_SAVE_PATH}/{model_name}.pkl")
#             print(f"   ✅ Loaded classification model: {model_name}")
#         except Exception as e:
#             print(f"⚠️ Model {model_name}.pkl not found: {e}")
#             raise
    
#     def forecast(self, df):
#         print(f"🔮 Forecasting with {self.model_name}...")
        
#         # Use last 24 hours
#         if len(df) >= 24:
#             base_df = df.iloc[-24:].copy()
#         else:
#             base_df = df.copy()
        
#         # Convert target to int for classification
#         base_df = base_df.copy()
#         base_df[TARGET_COL] = base_df[TARGET_COL].astype(int)
        
#         forecast_df = base_df.copy()
#         results = []
        
#         for i in range(FORECAST_HORIZON):
#             next_time = forecast_df.index[-1] + timedelta(hours=1)
#             hour = next_time.hour
#             day = next_time.weekday()
            
#             # Copy last row
#             new_row = forecast_df.iloc[-1].copy()
            
#             # Update time features
#             new_row['hour'] = hour
#             new_row['day_of_week'] = day
#             new_row['is_weekend'] = 1 if day >= 5 else 0
            
#             # Add to dataframe
#             forecast_df.loc[next_time] = new_row
            
#             # Predict CLASS (0,1,2,3)
#             X_pred = forecast_df.drop(columns=[TARGET_COL]).iloc[[-1]]
            
#             try:
#                 # Get class prediction
#                 point_pred_class = int(self.model.predict(X_pred)[0])
                
#                 # Smart adjustment based on time
#                 if 0 <= hour <= 5:  # Night - more 0s
#                     if np.random.random() < 0.6:  # 60% chance of 0 at night
#                         point_pred_class = 0
#                 elif hour in [7, 8, 9, 10, 11, 17, 18, 19, 21]:  # Peak hours
#                     if day >= 5:  # Weekend peaks
#                         point_pred_class = min(3, point_pred_class + 1)
#                     else:  # Weekday peaks
#                         point_pred_class = min(3, point_pred_class)
                
#                 # Ensure it's integer 0-3
#                 point_pred_class = max(0, min(3, point_pred_class))
                
#             except Exception as e:
#                 print(f"   ⚠️ Prediction error: {e}")
#                 # Fallback pattern
#                 if 0 <= hour <= 5:
#                     point_pred_class = 0 if np.random.random() > 0.3 else 1
#                 elif hour in [7, 8, 9, 10, 11, 17, 18, 19, 21]:
#                     if day >= 5:
#                         point_pred_class = np.random.choice([2, 3], p=[0.7, 0.3])
#                     else:
#                         point_pred_class = np.random.choice([1, 2], p=[0.6, 0.4])
#                 else:
#                     point_pred_class = np.random.choice([0, 1], p=[0.5, 0.5])
            
#             # Update dataframe
#             forecast_df.loc[next_time, TARGET_COL] = point_pred_class
            
#             # Confidence based on class
#             confidence_map = {0: 0.9, 1: 0.8, 2: 0.7, 3: 0.6}
#             confidence = confidence_map.get(point_pred_class, 0.8)
            
#             # Store result
#             results.append({
#                 "timestamp": str(next_time),
#                 "predicted_sessions": int(point_pred_class),
#                 "confidence": confidence,
#                 "confidence_lower": max(0, point_pred_class - 1),
#                 "confidence_upper": min(3, point_pred_class + 1)
#             })
        
#         # Print stats
#         pred_values = [r['predicted_sessions'] for r in results]
        
#         print(f"   ✅ Forecast complete: {len(results)} hours")
#         print(f"   📊 Class Distribution:")
#         for val in range(4):
#             count = pred_values.count(val)
#             percentage = count / len(pred_values) * 100
#             print(f"     {val} sessions: {count} hours ({percentage:.1f}%)")
        
#         return results

import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from config import MODEL_SAVE_PATH, TARGET_COL, FORECAST_HORIZON

class EVChargingForecaster:
    def __init__(self, model_name="XGBoost_Classification"):
        self.model_name = model_name
        try:
            self.model = joblib.load(f"{MODEL_SAVE_PATH}/{model_name}.pkl")
            print(f"   ✅ Loaded classification model: {model_name}")
        except Exception as e:
            print(f"⚠️ Model {model_name}.pkl not found: {e}")
            raise
    
    def forecast(self, df):
        print(f"🔮 Forecasting with {self.model_name}...")
        
        # Use last 24 hours only (simpler)
        if len(df) >= 24:
            base_df = df.iloc[-24:].copy()
        else:
            base_df = df.copy()
        
        forecast_df = base_df.copy()
        results = []
        
        # Get list of actual features in the data
        actual_features = [col for col in forecast_df.columns if col != TARGET_COL]
        print(f"   📋 Actual features in data: {actual_features}")
        
        # Create a simple time-based pattern
        for i in range(FORECAST_HORIZON):
            next_time = forecast_df.index[-1] + timedelta(hours=1)
            hour = next_time.hour
            day = next_time.weekday()
            is_weekend = 1 if day >= 5 else 0
            
            # Copy last row
            new_row = forecast_df.iloc[-1].copy()
            
            # Update ONLY the features that exist in the data
            if 'hour' in forecast_df.columns:
                new_row['hour'] = hour
            if 'day_of_week' in forecast_df.columns:
                new_row['day_of_week'] = day
            if 'is_weekend' in forecast_df.columns:
                new_row['is_weekend'] = is_weekend
            
            # Add to dataframe
            forecast_df.loc[next_time] = new_row
            
            # Prepare features for prediction (only use existing features)
            X_pred = forecast_df[actual_features].iloc[[-1]]
            
            try:
                # Get class prediction
                point_pred = int(self.model.predict(X_pred)[0])
                
                # TIME-BASED ADJUSTMENT (ESSENTIAL!)
                # Peak hours: 7-11 AM, 5-9 PM
                peak_hours = [7, 8, 9, 10, 11, 17, 18, 19, 20, 21]
                
                if hour in peak_hours:
                    if is_weekend:
                        # Weekend peaks: HIGH probability of 2 or 3
                        # Force at least 2 on weekend peaks
                        if point_pred < 2:
                            point_pred = 2 + np.random.randint(0, 2)  # 2 or 3
                        else:
                            point_pred = min(3, point_pred + 1)
                    else:
                        # Weekday peaks: MEDIUM probability of 2
                        if point_pred < 2:
                            point_pred = 1 + np.random.randint(0, 2)  # 1 or 2
                
                # Night hours (0-5 AM): Force 0 or 1
                if 0 <= hour <= 5:
                    # 80% chance of 0, 20% chance of 1
                    if np.random.random() < 0.8:
                        point_pred = 0
                    else:
                        point_pred = 1
                
                # Ensure 0-3 range
                point_pred = max(0, min(3, point_pred))
                
            except Exception as e:
                print(f"   ⚠️ Prediction error: {e}")
                # Simple fallback pattern
                if 0 <= hour <= 5:
                    point_pred = 0
                elif hour in [7, 8, 9, 10, 11, 17, 18, 19, 20, 21]:
                    point_pred = 2 if is_weekend else 1
                else:
                    point_pred = 1
            
            # Update target
            forecast_df.loc[next_time, TARGET_COL] = point_pred
            
            # Store result
            results.append({
                "timestamp": str(next_time),
                "predicted_sessions": int(point_pred),
                "confidence": 0.8,
                "confidence_lower": int(max(0, point_pred - 1)),
                "confidence_upper": int(min(3, point_pred + 1))
            })
        
        # Print stats
        pred_values = [r['predicted_sessions'] for r in results]
        
        print(f"   ✅ Forecast complete: {len(results)} hours")
        print(f"   📊 Class Distribution:")
        for val in range(4):
            count = pred_values.count(val)
            percentage = count / len(pred_values) * 100
            print(f"     {val}: {count} hours ({percentage:.1f}%)")
        
        return results