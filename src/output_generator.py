import json
import pandas as pd
from datetime import datetime

class OutputGenerator:
    
    @staticmethod
    def generate_backend_json(forecast_results, model_name="Model"):
        
        df = pd.DataFrame(forecast_results)
        
        # टाइमस्टैम्प प्रोसेस
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
        else:
            return {"error": "No timestamp"}
        
        # Sessions प्रोसेस
        if 'predicted_sessions' in df.columns:
            df['predicted_sessions'] = pd.to_numeric(df['predicted_sessions'], errors='coerce')
            df['predicted_sessions'] = df['predicted_sessions'].fillna(0).astype(int)
        else:
            return {"error": "No predicted_sessions"}
        
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "model": model_name,
                "total_hours": len(df),
                "version": "1.0"
            },
            "hourly_forecasts": []
        }
        
        for _, row in df.iterrows():
            demand = int(row['predicted_sessions'])
            
            # डिमांड लेवल
            if demand >= 3:
                level = "HIGH"
                multiplier = 1.3
                suggestion = "Consider surge pricing"
            elif demand == 0:
                level = "LOW"
                multiplier = 0.8
                suggestion = "Offer discounts"
            else:
                level = "MEDIUM"
                multiplier = 1.0
                suggestion = "Normal pricing"
            
            forecast_item = {
                "timestamp": row['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                "predicted_demand": demand,
                "confidence_interval": {
                    "lower": int(row.get('confidence_lower', max(0, demand - 1))),
                    "upper": int(row.get('confidence_upper', demand + 1))
                },
                "pricing": {
                    "demand_level": level,
                    "price_multiplier": multiplier,
                    "suggested_price_per_kwh": round(10.0 * multiplier, 2)
                },
                "load_management": {
                    "expected_load_kw": round(demand * 7.4, 1),
                    "recommendation": suggestion,
                    "available_chargers": max(0, 10 - demand)
                }
            }
            output["hourly_forecasts"].append(forecast_item)
        
        return output
    
    @staticmethod
    def generate_frontend_json(forecast_results):
        
        df = pd.DataFrame(forecast_results)
        
        # डेटा प्रोसेस
        if 'timestamp' not in df.columns:
            return {"error": "No timestamp"}
        
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        
        # हीटमैप डेटा
        heatmap_data = {}
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
        
        for day in days_order:
            heatmap_data[day] = {}
            for hour in range(24):
                hour_data = df[(df['day'] == day) & (df['hour'] == hour)]
                if len(hour_data) > 0:
                    avg_demand = hour_data['predicted_sessions'].mean()
                    heatmap_data[day][str(hour)] = round(avg_demand, 1)
                else:
                    heatmap_data[day][str(hour)] = 0.0
        
        # बेस्ट टाइम्स (सबसे कम डिमांड)
        if 'predicted_sessions' in df.columns:
            cheapest_times = df.nsmallest(5, 'predicted_sessions')
        else:
            cheapest_times = df.head(5)
        
        best_times = []
        for _, row in cheapest_times.iterrows():
            demand = row.get('predicted_sessions', 0)
            level = "LOW" if demand <= 1 else "MEDIUM" if demand <= 2 else "HIGH"
            
            best_times.append({
                "day": row['day'],
                "hour": int(row['hour']),
                "date": row['date_str'],
                "expected_demand": int(demand),
                "price_indicator": level
            })
        
        # सारांश
        avg_demand = df['predicted_sessions'].mean() if 'predicted_sessions' in df.columns else 0
        peak_demand = df['predicted_sessions'].max() if 'predicted_sessions' in df.columns else 0
        
        output = {
            "heatmap": heatmap_data,
            "best_times_to_charge": best_times,
            "summary": {
                "avg_demand": round(avg_demand, 1),
                "peak_demand": int(peak_demand),
                "total_forecast_hours": len(df)
            }
        }
        
        return output
    
    @staticmethod
    def save_json_outputs(backend_json, frontend_json, model_name="Model"):
        """JSON आउटपुट्स सेव करें"""
        import os
        
        os.makedirs("outputs", exist_ok=True)
        
        # बैकएंड JSON
        backend_file = f"outputs/backend_{model_name}.json"
        with open(backend_file, 'w') as f:
            json.dump(backend_json, f, indent=2)
        
        # फ्रंटएंड JSON
        frontend_file = f"outputs/frontend_{model_name}.json"
        with open(frontend_file, 'w') as f:
            json.dump(frontend_json, f, indent=2)
        
        print(f"✅ Backend JSON: {backend_file}")
        print(f"✅ Frontend JSON: {frontend_file}")
        
        return backend_file, frontend_file
    
