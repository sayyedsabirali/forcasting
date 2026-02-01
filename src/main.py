import os
import json
import pandas as pd
import numpy as np
from data_loader import DataLoader
from model_trainer import ModelTrainer
from forecaster import EVChargingForecaster
from visualizer import Visualizer
from output_generator import OutputGenerator
from config import (
    TARGET_COL, OUTPUT_PATH, PLOTS_PATH, MODEL_SAVE_PATH,
    FORECAST_HORIZON, BASE_DIR
)


print("\n" + "="*70)
print("🚀 EV CHARGING DEMAND FORECASTING PIPELINE")
print("="*70 + "\n")
print("\n" + "="*70)
print("🚀 EV CHARGING DEMAND FORECASTING PIPELINE")
print("="*70 + "\n")

# 1. Load data
print("📥 STEP 1: Loading data...")
df = DataLoader().load_data()
print(f"   ✅ Data shape: {df.shape}")

# 2. Train-test split
print("\n✂️ STEP 2: Train-test split (80/20)...")
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

X_train = train.drop(columns=[TARGET_COL])
y_train = train[TARGET_COL]
X_test = test.drop(columns=[TARGET_COL])
y_test = test[TARGET_COL]

print(f"   Train: {len(X_train)} samples")
print(f"   Test: {len(X_test)} samples")

# # 3. Train models
# print("\n🤖 STEP 3: Training models...")
# trainer = ModelTrainer()
# trained_models, results = trainer.train_all(X_train, y_train, X_test, y_test)

# # 4. Visualizations
# print("\n📊 STEP 4: Creating visualizations...")
# viz = Visualizer()
# viz.plot_feature_importance(trained_models, X_train)
# viz.plot_model_comparison(results)

# # Actual vs Predicted
# predictions_dict = {name: results[name]['predictions'] for name in results.keys()}
# viz.plot_actual_vs_predicted(y_test.values, predictions_dict, list(results.keys()))

# # 5. Forecasting
# print("\n🔮 STEP 5: Generating 7-day forecast...")
# best_model_name = min(results, key=lambda x: results[x]['rmse'])
# print(f"   🏆 Best model: {best_model_name}")

# forecaster = EVChargingForecaster(best_model_name)
# forecast_results = forecaster.forecast(df)

# # 6. Plot forecast
# print("   🎨 Creating forecast plots...")
# viz.plot_forecast(forecast_results, best_model_name)

# print("\n" + "="*70)
# print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
# print("="*70 + "\n")

# STEP 3 में:
print("\n🤖 STEP 3: Training CLASSIFICATION models...")

# y_train और y_test को integer में convert करो
y_train = y_train.astype(int)
y_test = y_test.astype(int)

print(f"   y_train classes: {sorted(y_train.unique())}")

# Use classification trainer
trainer = ModelTrainer()
trained_models, results = trainer.train_all(X_train, y_train, X_test, y_test)

# STEP 5 में:
print("\n🔮 STEP 5: Generating 7-day forecast...")
best_model_name = min(results, key=lambda x: results[x]['rmse'])
print(f"   🏆 Using best model: {best_model_name}")

# Ensure target is integer
df[TARGET_COL] = df[TARGET_COL].astype(int)

forecaster = EVChargingForecaster(best_model_name)
forecast_results = forecaster.forecast(df)
viz = Visualizer()
viz.plot_forecast(forecast_results, best_model_name)
# 7. JSON आउटपुट जेनरेट
print("\n💾 STEP 6: Generating JSON outputs...")
os.makedirs(OUTPUT_PATH, exist_ok=True)

try:
    # बैकएंड के लिए JSON
    backend_json = OutputGenerator.generate_backend_json(forecast_results)
    with open(f"{OUTPUT_PATH}/backend_forecast.json", "w", encoding='utf-8') as f:
        json.dump(backend_json, f, indent=2, ensure_ascii=False)
    
    # फ्रंटएंड के लिए JSON
    frontend_json = OutputGenerator.generate_frontend_json(forecast_results)
    with open(f"{OUTPUT_PATH}/frontend_forecast.json", "w", encoding='utf-8') as f:
        json.dump(frontend_json, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ JSON outputs saved to {OUTPUT_PATH}/")
    
except Exception as e:
    print(f"   ❌ JSON generation error: {e}")

# 8. सारांश प्रिंट करें
print("\n" + "="*70)
print("📊 FORECAST SUMMARY")
print("="*70)

forecast_df = pd.DataFrame(forecast_results)
forecast_df['timestamp'] = pd.to_datetime(forecast_df['timestamp'])
forecast_df['date'] = forecast_df['timestamp'].dt.date
forecast_df['hour'] = forecast_df['timestamp'].dt.hour
forecast_df['day_name'] = forecast_df['timestamp'].dt.day_name()

print(f"\n📅 Forecast Period: {forecast_df['timestamp'].iloc[0]} to {forecast_df['timestamp'].iloc[-1]}")
print(f"📈 Avg Sessions/Hour: {forecast_df['predicted_sessions'].mean():.2f}")
print(f"🔥 Peak Sessions: {forecast_df['predicted_sessions'].max():.2f}")
print(f"💧 Minimum Sessions: {forecast_df['predicted_sessions'].min():.2f}")

# Prediction distribution
print(f"\n📊 Forecast Distribution:")
for val in [0, 1, 2, 3]:
    count = (forecast_df['predicted_sessions'] == val).sum()
    percentage = (count / len(forecast_df)) * 100
    print(f"   {val} sessions: {count} hours ({percentage:.1f}%)")

# दैनिक सारांश
print("\n📅 DAILY SUMMARY:")
daily_summary = forecast_df.groupby('date').agg({
    'predicted_sessions': ['sum', 'mean', 'max']
}).round(2)

daily_summary.columns = ['Total', 'Avg per Hour', 'Peak']
print(daily_summary.to_string())

# सर्वोत्तम चार्जिंग समय (Lowest demand)
print("\n💡 BEST TIMES TO CHARGE (Lowest Demand):")
low_demand_times = forecast_df[forecast_df['predicted_sessions'] <= 1]
if len(low_demand_times) > 0:
    low_demand_times = low_demand_times.nsmallest(5, 'predicted_sessions')[['timestamp', 'predicted_sessions']]
    for idx, row in low_demand_times.iterrows():
        time_str = pd.to_datetime(row['timestamp']).strftime('%A, %d %b %Y %H:00')
        print(f"   {time_str}: {row['predicted_sessions']:.0f} sessions")
else:
    print("   No low demand periods found")

print("\n" + "="*70)
print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70 + "\n")

print("📁 Output Files:")
print(f"   📊 Plots: {PLOTS_PATH}/")
print(f"   📁 Data: {OUTPUT_PATH}/")
print(f"   📄 backend_forecast.json - For dynamic pricing engine")
print(f"   📄 frontend_forecast.json - For UI/display")
print(f"   📄 heatmap_data.csv - For demand heatmap\n")