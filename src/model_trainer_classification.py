import os
import joblib
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
import numpy as np
from config import MODEL_SAVE_PATH, RANDOM_STATE

class ModelTrainerClassification:

    def train_all(self, X_train, y_train, X_test, y_test):
        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
        
        # Convert to classification: 0, 1, 2, 3 sessions
        y_train_class = y_train.astype(int)
        y_test_class = y_test.astype(int)
        
        print(f"\n📊 CLASSIFICATION TARGET DISTRIBUTION:")
        for i in range(4):
            count = (y_train_class == i).sum()
            print(f"  {i} sessions: {count} samples ({count/len(y_train_class)*100:.1f}%)")
        
        # Classification models
        xgb_params = {
            'n_estimators': 1000,
            'max_depth': 10,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': RANDOM_STATE,
            'objective': 'multi:softmax',
            'num_class': 4,
            'verbosity': 0
        }
        
        rf_params = {
            'n_estimators': 500,
            'max_depth': 20,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'random_state': RANDOM_STATE,
            'class_weight': 'balanced',  # IMPORTANT!
            'verbose': 0
        }
        
        models_dict = {
            "XGBoost_Classification": xgb.XGBClassifier(**xgb_params),
            "RandomForest_Classification": RandomForestClassifier(**rf_params)
        }
        
        trained_models = {}
        results = {}
        
        for name, model in models_dict.items():
            print(f"\n🎯 Training {name}...")
            
            model.fit(X_train, y_train_class)
            
            # Predict classes
            y_pred_class = model.predict(X_test)
            
            # Convert back to "regression-like" for comparison
            y_pred_reg = y_pred_class.astype(float)
            
            # Metrics
            mae = mean_absolute_error(y_test, y_pred_reg)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
            accuracy = accuracy_score(y_test_class, y_pred_class)
            f1 = f1_score(y_test_class, y_pred_class, average='weighted')
            
            print(f"   ✅ Training complete")
            print(f"   📊 Metrics - RMSE: {rmse:.3f}, MAE: {mae:.3f}")
            print(f"   🎯 Classification - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            
            # Check distribution
            print(f"   🔍 Predicted class distribution:")
            for i in range(4):
                count = (y_pred_class == i).sum()
                print(f"      {i} sessions: {count} ({count/len(y_pred_class)*100:.1f}%)")
            
            # Save model
            joblib.dump(model, f"{MODEL_SAVE_PATH}/{name}.pkl")
            
            # Store results
            trained_models[name] = model
            results[name] = {
                'mae': mae,
                'rmse': rmse,
                'accuracy': accuracy,
                'f1': f1,
                'predictions': y_pred_reg,
                'predictions_class': y_pred_class
            }
        
        return trained_models, results