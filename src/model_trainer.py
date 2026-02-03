import os
import joblib
import xgboost as xgb
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from config import MODEL_SAVE_PATH, RANDOM_STATE


class ModelTrainer:

    def train_all(self, X_train, y_train, X_test, y_test):

        os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

        print("\n📊 Training class distribution:")
        for i in range(4):
            count = (y_train == i).sum()
            print(f"   Class {i}: {count} samples ({count/len(y_train)*100:.1f}%)")

        # 🔥 CLASS WEIGHTS (important)
        class_weight_dict = {
            0: 1,
            1: 2,
            2: 6,
            3: 16
        }
        rf_model = RandomForestClassifier(
            n_estimators=400,
            max_depth=16,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight=class_weight_dict
        )
        sample_weights = y_train.map(class_weight_dict)

        xgb_model = xgb.XGBClassifier(
            n_estimators=600,
            max_depth=9,
            learning_rate=0.05,
            objective='multi:softprob',
            num_class=4,
            random_state=RANDOM_STATE,
            eval_metric='mlogloss'
        )

        models_dict = {
            "RandomForest_Classification": rf_model,
            "XGBoost_Classification": xgb_model
        }

        trained_models = {}
        results = {}

        for name, model in models_dict.items():

            print(f"\n🚀 Training {name}...")

            if "XGBoost" in name:
                model.fit(X_train, y_train, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)

            print(f"   🎯 Accuracy: {acc:.4f}")
            print("\n📄 Classification Report:")
            print(classification_report(y_test, y_pred))

            joblib.dump(model, f"{MODEL_SAVE_PATH}/{name}.pkl")

            trained_models[name] = model
            results[name] = {"accuracy": acc}

        return trained_models, results
