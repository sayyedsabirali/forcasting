import json
import pandas as pd
from datetime import datetime

class OutputGenerator:
    
    @staticmethod
    def generate_backend_json(forecast_results):
        """बैकएंड के लिए JSON"""
        output = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_hours": len(forecast_results),
                "version": "1.0"
            },
            "hourly_forecasts": []
        }
        
        for item in forecast_results:
            demand = item["predicted_sessions"]
            
            # डायनामिक प्राइसिंग लेवल
            if demand >= 8:
                demand_level = "HIGH"
                price_multiplier = 1.3
                suggestion = "Consider surge pricing"
            elif demand <= 3:
                demand_level = "LOW"
                price_multiplier = 0.8
                suggestion = "Offer discounts"
            else:
                demand_level = "MEDIUM"
                price_multiplier = 1.0
                suggestion = "Normal pricing"
            
            forecast_item = {
                "timestamp": item["timestamp"],
                "predicted_demand": round(demand, 2),
                "confidence_interval": {
                    "lower": round(item.get("confidence_lower", demand * 0.8), 2),
                    "upper": round(item.get("confidence_upper", demand * 1.2), 2)
                },
                "pricing": {
                    "demand_level": demand_level,
                    "price_multiplier": price_multiplier,
                    "suggested_price_per_kwh": round(10.0 * price_multiplier, 2)
                },
                "load_management": {
                    "expected_load_kw": round(demand * 7.4, 2),
                    "recommendation": suggestion
                }
            }
            output["hourly_forecasts"].append(forecast_item)
        
        return output
    
    @staticmethod
    def generate_frontend_json(forecast_results):
        """फ्रंटएंड/UI के लिए JSON"""
        df = pd.DataFrame(forecast_results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.day_name()
        df['date_str'] = df['timestamp'].dt.strftime('%Y-%m-%d')
        
        # हीटमैप डेटा
        heatmap_data = {}
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for day in days_order:
            day_data = df[df['day'] == day]
            if len(day_data) > 0:
                heatmap_data[day] = {}
                for hour in range(24):
                    hour_data = day_data[day_data['hour'] == hour]
                    if len(hour_data) > 0:
                        heatmap_data[day][str(hour)] = round(hour_data['predicted_sessions'].mean(), 1)
                    else:
                        heatmap_data[day][str(hour)] = 0.0
        
        # बेस्ट टाइम टू चार्ज (लोएस्ट प्राइस)
        df['price_multiplier'] = df['predicted_sessions'].apply(
            lambda x: 0.8 if x <= 3 else (1.3 if x >= 8 else 1.0)
        )
        cheapest_times = df.nsmallest(5, 'price_multiplier')
        
        best_times = []
        for _, row in cheapest_times.iterrows():
            best_times.append({
                "day": row['day'],
                "hour": int(row['hour']),
                "date": row['date_str'],
                "expected_demand": round(row['predicted_sessions'], 1),
                "price_indicator": "LOW"
            })
        
        output = {
            "heatmap": heatmap_data,
            "best_times_to_charge": best_times,
            "summary": {
                "avg_demand": round(df['predicted_sessions'].mean(), 1),
                "peak_demand": round(df['predicted_sessions'].max(), 1),
                "total_forecast_hours": len(df)
            }
        }
        
        return output