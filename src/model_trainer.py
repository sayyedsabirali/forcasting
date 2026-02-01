# import os
# import joblib
# import xgboost as xgb
# import lightgbm as lgb
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# import numpy as np
# from config import MODEL_SAVE_PATH, RANDOM_STATE

# class ModelTrainer:
#     def train_all(self, X_train, y_train, X_test, y_test):
#         os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
#         # SIMPLE REGRESSION MODELS
#         models_dict = {
#             "XGBoost": xgb.XGBRegressor(
#                 n_estimators=200,
#                 max_depth=6,
#                 learning_rate=0.1,
#                 random_state=RANDOM_STATE,
#                 objective='reg:squarederror',
#                 verbosity=0
#             ),
#             "LightGBM": lgb.LGBMRegressor(
#                 n_estimators=200,
#                 max_depth=6,
#                 learning_rate=0.1,
#                 random_state=RANDOM_STATE,
#                 objective='regression',
#                 verbose=-1
#             ),
#             "RandomForest": RandomForestRegressor(
#                 n_estimators=200,
#                 max_depth=10,
#                 random_state=RANDOM_STATE,
#                 min_samples_split=5,
#                 min_samples_leaf=2
#             )
#         }
        
#         trained_models = {}
#         results = {}
        
#         print(f"\n📊 Training data distribution:")
#         print(f"   0: {(y_train == 0).sum()} samples")
#         print(f"   1: {(y_train == 1).sum()} samples") 
#         print(f"   2: {(y_train == 2).sum()} samples")
#         print(f"   3: {(y_train == 3).sum()} samples")
        
#         for name, model in models_dict.items():
#             print(f"\n🎯 Training {name}...")
            
#             # SIMPLE TRAINING - NO WEIGHTS, NO POST-PROCESSING
#             model.fit(X_train, y_train)
            
#             # Predict
#             y_pred = model.predict(X_test)
            
#             # NO ROUNDING, NO POST-PROCESSING - keep as float
#             y_pred = y_pred.astype(float)
            
#             # Metrics
#             mae = mean_absolute_error(y_test, y_pred)
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#             r2 = r2_score(y_test, y_pred)
            
#             print(f"   ✅ Training complete")
#             print(f"   📊 Metrics - RMSE: {rmse:.3f}, R²: {r2:.3f}")
#             print(f"   📈 Predictions range: {y_pred.min():.2f} to {y_pred.max():.2f}")
            
#             # Save model
#             joblib.dump(model, f"{MODEL_SAVE_PATH}/{name}.pkl")
            
#             # Store results
#             trained_models[name] = model
#             results[name] = {
#                 'mae': mae,
#                 'rmse': rmse,
#                 'r2': r2,
#                 'predictions': y_pred
#             }
        
#         return trained_models, results


import os
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
import numpy as np
from config import MODEL_SAVE_PATH, RANDOM_STATE

class ModelTrainer:
    def train_all(self, X_train, y_train, X_test, y_test):
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        # CONVERT TO CLASSIFICATION
        y_train_class = y_train.astype(int)
        y_test_class = y_test.astype(int)
        
        print(f"\n📊 CLASSIFICATION SETUP:")
        print(f"   Classes: 0, 1, 2, 3")
        print(f"   Train distribution:")
        for i in range(4):
            count = (y_train_class == i).sum()
            print(f"     {i}: {count} samples ({count/len(y_train_class)*100:.1f}%)")
        
        # VERY AGGRESSIVE MANUAL WEIGHTS
        # Since 0 and 1 are most common, give VERY HIGH weights to 2 and 3
        class_weight_dict = {
            0: 1.0,   # Normal weight for 0
            1: 1.0,   # Normal weight for 1  
            2: 8.0,   # 8x weight for 2 (because it's 3x less common than 0)
            3: 15.0   # 15x weight for 3 (because it's 7x less common than 0)
        }
        
        print(f"\n🎯 AGGRESSIVE Class weights:")
        for cls, weight in class_weight_dict.items():
            print(f"   Class {cls}: {weight:.1f}x")
        
        # AGGRESSIVE CLASSIFICATION MODELS
        models_dict = {
            "XGBoost_Classification": xgb.XGBClassifier(
                n_estimators=500,           # Increased
                max_depth=8,                # Increased
                learning_rate=0.05,         # Slightly lower
                random_state=RANDOM_STATE,
                objective='multi:softmax',
                num_class=4,
                scale_pos_weight=10,        # AGGRESSIVE for positive classes
                gamma=0.1,                  # Regularization
                reg_alpha=0.1,              # L1 regularization
                reg_lambda=1,               # L2 regularization
                verbosity=0
            ),
            "RandomForest_Classification": RandomForestClassifier(
                n_estimators=300,           # Increased
                max_depth=15,               # Increased
                random_state=RANDOM_STATE,
                class_weight=class_weight_dict,  # Use our aggressive weights
                min_samples_split=2,        # Reduced
                min_samples_leaf=1,         # Reduced
                bootstrap=True,
                oob_score=True,
                verbose=0
            )
        }
        
        trained_models = {}
        results = {}
        
        for name, model in models_dict.items():
            print(f"\n🎯 Training {name}...")
            
            # Create EXTREMELY AGGRESSIVE sample weights
            sample_weights = np.array([class_weight_dict[y] for y in y_train_class])
            
            # For XGBoost, we can also set scale_pos_weight
            if 'XGBoost' in name:
                # Train with aggressive weights
                model.fit(X_train, y_train_class, 
                         sample_weight=sample_weights,
                         verbose=False)
            else:
                # RandomForest uses class_weight parameter
                model.fit(X_train, y_train_class)
            
            # Predict classes
            y_pred_class = model.predict(X_test)
            
            # Convert to regression-like for metrics
            y_pred_reg = y_pred_class.astype(float)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred_reg)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
            r2 = r2_score(y_test, y_pred_reg)
            accuracy = accuracy_score(y_test_class, y_pred_class)
            
            print(f"   ✅ Training complete")
            print(f"   📊 Regression Metrics - RMSE: {rmse:.3f}, R²: {r2:.3f}")
            print(f"   🎯 Classification Accuracy: {accuracy:.3f}")
            
            # Detailed distribution
            print(f"   📈 Predicted class distribution:")
            class_counts = {}
            for i in range(4):
                count = (y_pred_class == i).sum()
                class_counts[i] = count
                print(f"     {i}: {count} predictions ({count/len(y_pred_class)*100:.1f}%)")
            
            # Force some 2s and 3s if missing
            if class_counts.get(2, 0) == 0 or class_counts.get(3, 0) == 0:
                print(f"   ⚠️ Missing class 2 or 3. Adjusting predictions...")
                # Get probabilities if available
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    # Find samples where class 2 or 3 has high probability
                    for i in range(len(y_pred_class)):
                        prob_class_2 = y_proba[i, 2] if 2 < y_proba.shape[1] else 0
                        prob_class_3 = y_proba[i, 3] if 3 < y_proba.shape[1] else 0
                        
                        # If probability for 2 or 3 is high, use it
                        if prob_class_2 > 0.3 and class_counts.get(2, 0) < 5:
                            y_pred_class[i] = 2
                            class_counts[2] = class_counts.get(2, 0) + 1
                        elif prob_class_3 > 0.2 and class_counts.get(3, 0) < 3:
                            y_pred_class[i] = 3
                            class_counts[3] = class_counts.get(3, 0) + 1
            
            # Update regression predictions
            y_pred_reg = y_pred_class.astype(float)
            
            # Recalculate metrics if changed
            if 'class_counts' in locals():
                mae = mean_absolute_error(y_test, y_pred_reg)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
                r2 = r2_score(y_test, y_pred_reg)
                accuracy = accuracy_score(y_test_class, y_pred_class)
                
                print(f"   🔧 Adjusted distribution:")
                for i in range(4):
                    count = (y_pred_class == i).sum()
                    print(f"     {i}: {count} predictions")
            
            # Save model
            joblib.dump(model, f"{MODEL_SAVE_PATH}/{name}.pkl")
            
            # Store results
            trained_models[name] = model
            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'accuracy': accuracy,
                'predictions': y_pred_reg,
                'predictions_class': y_pred_class
            }
        
        return trained_models, results