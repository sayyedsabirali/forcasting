import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.dates import DateFormatter
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tabulate import tabulate
import warnings
import json
warnings.filterwarnings('ignore')


class Visualizer:
    
    def __init__(self, plots_path="plots"):
        self.PLOTS_PATH = plots_path
        os.makedirs(self.PLOTS_PATH, exist_ok=True)
        self.setup_plot_style()
    
    def setup_plot_style(self):
        """प्लॉट स्टाइल सेट करें"""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['font.family'] = 'DejaVu Sans'
    
    def plot_feature_importance(self, models, X_train=None, feature_names=None):
        """फीचर इंपोर्टेंस प्लॉट करें"""
        
        # फीचर नेम्स हासिल करें
        if X_train is not None and hasattr(X_train, 'columns'):
            feature_names = list(X_train.columns)
        elif feature_names is None:
            print("⚠️ Feature names not provided")
            return
        
        # डुप्लीकेट हटाएं
        unique_features = []
        seen = set()
        for f in feature_names:
            if f not in seen:
                seen.add(f)
                unique_features.append(f)
        feature_names = unique_features
        
        model_names = list(models.keys())
        n_models = len(model_names)
        
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 8))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(model_names):
            model = models[model_name]
            ax = axes[idx]
            
            try:
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importances = np.abs(model.coef_)
                else:
                    print(f"{model_name} - No feature importances")
                    continue
                
                # टॉप 10 फीचर्स
                top_n = min(10, len(importances))
                indices = np.argsort(importances)[-top_n:]
                
                ax.barh(range(len(indices)), importances[indices], 
                       color=plt.cm.viridis(np.linspace(0.3, 0.9, len(indices))))
                
                ax.set_yticks(range(len(indices)))
                
                # फीचर नेम्स फॉर्मेट करें
                names = []
                for i in indices:
                    name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                    name = name.replace('_', ' ').title()
                    if len(name) > 25:
                        name = name[:22] + "..."
                    names.append(name)
                
                ax.set_yticklabels(names, fontsize=9)
                ax.set_xlabel('Importance Score', fontsize=10)
                ax.set_title(f'{model_name}\nTop Features', fontsize=12)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')
                
            except Exception as e:
                print(f"Error in {model_name}: {e}")
        
        plt.suptitle('Feature Importance Comparison', fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{self.PLOTS_PATH}/feature_importance.png", dpi=300)
        plt.show()
        
        # अलग-अलग फाइल्स
        self.save_individual_feature_plots(models, feature_names)
    
    def save_individual_feature_plots(self, models, feature_names):
        """हर मॉडल के लिए अलग फीचर इंपोर्टेंस प्लॉट सेव करें"""
        for model_name, model in models.items():
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                else:
                    continue
                
                # टॉप 15 फीचर्स
                top_n = min(15, len(importances))
                indices = np.argsort(importances)[-top_n:]
                
                colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(indices)))
                bars = ax.barh(range(len(indices)), importances[indices], 
                              color=colors, edgecolor='black')
                
                ax.set_yticks(range(len(indices)))
                
                # फीचर नेम्स
                display_names = []
                for i in indices:
                    name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
                    name = name.replace('_', ' ').title()
                    display_names.append(name)
                
                ax.set_yticklabels(display_names, fontsize=10)
                ax.set_xlabel('Importance Score', fontsize=12)
                ax.set_title(f'Feature Importance - {model_name}', fontsize=14)
                ax.invert_yaxis()
                ax.grid(True, alpha=0.3, axis='x')
                
                plt.tight_layout()
                plt.savefig(f"{self.PLOTS_PATH}/feature_importance_{model_name}.png", dpi=300)
                plt.close(fig)
                
            except Exception as e:
                print(f"  Skipping {model_name}: {e}")
    
    def _plot_metric_bar(self, ax, model_names, values, color, title, note):
        """मेट्रिक बार प्लॉट हेल्पर"""
        bars = ax.bar(model_names, values, color=color, edgecolor='black', alpha=0.8)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'{title}\n{note}', fontsize=13)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    
    def plot_forecast(self, forecast_results, model_name="Model"):
        df = pd.DataFrame(forecast_results)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
        else:
            print("❌ No timestamp column")
            return
        if 'predicted_sessions' in df.columns:
            df['predicted_sessions'] = pd.to_numeric(df['predicted_sessions'], errors='coerce')
            df['predicted_sessions'] = df['predicted_sessions'].fillna(0).astype(int)
        else:
            print("❌ No predicted_sessions column")
            return
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        self._plot_forecast_line(axes[0], df, model_name)
        self._plot_forecast_heatmap(axes[1], df, model_name)
        
        plt.tight_layout()
        plt.savefig(f"{self.PLOTS_PATH}/7day_forecast_{model_name}.png", dpi=300)
        plt.show()
    
    def _plot_forecast_line(self, ax, df, model_name):
        """फोरकास्ट लाइन प्लॉट"""
        # ax.plot(df['timestamp'], df['predicted_sessions'], 
        #        marker='o', markersize=5, linewidth=2, color='#2E86AB')
        # Line plot की जगह bar chart
        ax.bar(df['timestamp'], df['predicted_sessions'])
        
        ax.set_xlabel('Date & Time', fontsize=12)
        ax.set_ylabel('Predicted Sessions', fontsize=12)
        ax.set_title(f'7-Day Forecast - {model_name}', fontsize=16)
        ax.grid(True, alpha=0.3)
        
        # X-टिक्स फॉर्मेट
        ax.xaxis.set_major_formatter(DateFormatter('%d %b\n%H:00'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Y-axis integer
        y_max = df['predicted_sessions'].max()
        ax.set_yticks(range(0, int(y_max) + 2))
    
    def _plot_forecast_heatmap(self, ax, df, model_name):
        """फोरकास्ट हीटमैप"""
        df['hour'] = df['timestamp'].dt.hour
        df['day_name'] = df['timestamp'].dt.day_name()
        
        # पिवट टेबल
        pivot_df = df.pivot_table(values='predicted_sessions', 
                                index='day_name', 
                                columns='hour', 
                                aggfunc='mean',
                                fill_value=0)
        
        # Round and fill
        pivot_df = pivot_df.round(0).astype(int)
        
        # Days order
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
        pivot_df = pivot_df.reindex(days_order)
        
        # Missing hours
        for h in range(24):
            if h not in pivot_df.columns:
                pivot_df[h] = 0
        pivot_df = pivot_df.sort_index(axis=1)
        
        # हीटमैप
        sns.heatmap(pivot_df, ax=ax, cmap='YlOrRd', annot=True, fmt='d',
                   cbar_kws={'label': 'Sessions'}, linewidths=0.5)
        
        ax.set_title('Weekly Heatmap', fontsize=16)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)
    

    def plot_7day_bar_chart(self, forecast_results, model_name="Model"):
    

        df = pd.DataFrame(forecast_results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['day_name'] = df['timestamp'].dt.day_name()
        
        # डेली एवरेज calculate करो
        daily_avg = df.groupby(['date', 'day_name']).agg({
            'predicted_sessions': 'mean',
            'confidence_lower': 'mean',
            'confidence_upper': 'mean'
        }).reset_index()
        
        # Sort by date
        daily_avg = daily_avg.sort_values('date')
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # X positions (0 से 6 तक - 7 days)
        x_pos = np.arange(len(daily_avg))
        
        # बार्स बनाओ
        bars = ax.bar(x_pos, daily_avg['predicted_sessions'], 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD166', 
                            '#06D6A0', '#118AB2', '#EF476F'],
                    edgecolor='black', alpha=0.8)
        
        # Confidence interval error bars
        ax.errorbar(x_pos, daily_avg['predicted_sessions'],
                    yerr=[daily_avg['predicted_sessions'] - daily_avg['confidence_lower'],
                        daily_avg['confidence_upper'] - daily_avg['predicted_sessions']],
                    fmt='none', ecolor='black', capsize=5, capthick=2)
        
        # X-axis labels
        labels = []
        for idx, row in daily_avg.iterrows():
            date_str = row['date'].strftime('%d %b')
            labels.append(f"{row['day_name'][:3]}\n{date_str}")
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=11, rotation=0)
        
        # Y-axis
        ax.set_ylim(bottom=0)
        ax.set_ylabel('Average Sessions per Hour', fontsize=12)
        
        # Title
        ax.set_title(f'7-Day EV Charging Demand Forecast\n({model_name})', 
                    fontsize=16, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=0.3, axis='y')
        
        # Values लिखो bars के ऊपर
        for bar, val in zip(bars, daily_avg['predicted_sessions']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(f"{self.PLOTS_PATH}/7day_bar_chart_{model_name}.png", dpi=300)
        plt.show()


    def plot_hourly_grouped_barchart(self, forecast_results, model_name="Model"):
        
        df = pd.DataFrame(forecast_results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour'] = df['timestamp'].dt.hour
        df['day_name'] = df['timestamp'].dt.day_name()
        
        # पिवट टेबल
        pivot_df = df.pivot_table(values='predicted_sessions',
                                index='hour',
                                columns='day_name',
                                aggfunc='mean')
        
        # Days order
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                    'Friday', 'Saturday', 'Sunday']
        pivot_df = pivot_df[days_order]
        
        # Grouped bar chart
        fig, ax = plt.subplots(figsize=(20, 10))
        
        # X positions
        x = np.arange(24)
        width = 0.12  # हर bar की width
        
        # Colors for each day
        colors = plt.cm.Set3(np.linspace(0, 1, 7))
        
        # हर दिन के लिए bars
        for i, day in enumerate(days_order):
            offset = (i - 3) * width  # Center align
            bars = ax.bar(x + offset, pivot_df[day], width,
                        label=day, color=colors[i], alpha=0.8)
        
        # Labels
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Predicted Sessions', fontsize=12)
        ax.set_title(f'Hourly Demand by Day of Week\n({model_name})',
                    fontsize=16, fontweight='bold')
        
        # X-ticks (0-23 hours)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45)
        
        # Legend
        ax.legend(title='Day of Week', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Grid
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f"{self.PLOTS_PATH}/hourly_grouped_bars_{model_name}.png", dpi=300)
        plt.show()

    def plot_forecast_with_confidence(self, forecast_results, model_name="Model"):
            
            df = pd.DataFrame(forecast_results)
            
            # टाइमस्टैम्प प्रोसेस करें
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
            else:
                print("❌ No timestamp column")
                return
            
            # Sessions और confidence intervals
            if 'predicted_sessions' in df.columns:
                df['predicted_sessions'] = pd.to_numeric(df['predicted_sessions'], errors='coerce')
                df['predicted_sessions'] = df['predicted_sessions'].fillna(0).astype(int)
            else:
                print("❌ No predicted_sessions column")
                return
            
            # Confidence intervals check
            if 'confidence_lower' not in df.columns or 'confidence_upper' not in df.columns:
                print("⚠️ No confidence interval data")
                return
            
            # Single plot with confidence intervals
            fig, ax = plt.subplots(figsize=(18, 8))
            
            # X positions
            x_pos = np.arange(len(df))
            bar_width = 0.6
            
            # Colors based on demand level
            colors = []
            for sessions in df['predicted_sessions']:
                if sessions >= 3:
                    colors.append('#FF6B6B')  # High demand - red
                elif sessions == 2:
                    colors.append('#FFD166')  # Medium demand - yellow
                elif sessions == 1:
                    colors.append('#4ECDC4')  # Low demand - teal
                else:
                    colors.append('#95E1D3')  # Very low - light teal
            
            # बार्स बनाओ with error bars
            bars = ax.bar(x_pos, df['predicted_sessions'], 
                        width=bar_width,
                        color=colors,
                        edgecolor='black',
                        alpha=0.8,
                        yerr=[df['predicted_sessions'] - df['confidence_lower'],
                            df['confidence_upper'] - df['predicted_sessions']],
                        capsize=3,
                        error_kw={'elinewidth': 1.5, 'ecolor': 'black', 'capthick': 1.5})
            
            # X-axis labels (every 12 hours)
            tick_positions = []
            tick_labels = []
            for i in range(0, len(df), 12):
                tick_positions.append(i)
                tick_labels.append(df['timestamp'].iloc[i].strftime('%d %b\n%H:00'))
            
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels, fontsize=10, rotation=45, ha='right')
            
            # Y-axis
            ax.set_ylim(bottom=0)
            ax.set_ylabel('Predicted Sessions', fontsize=12)
            ax.set_yticks(range(0, 5))
            
            # Title
            ax.set_title(f'7-Day EV Charging Demand Forecast with Confidence Intervals\n({model_name})', 
                        fontsize=16, fontweight='bold')
            
            # Grid
            ax.grid(True, alpha=0.3, axis='y')
            
            # Legend for colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#95E1D3', edgecolor='black', label='0 Sessions'),
                Patch(facecolor='#4ECDC4', edgecolor='black', label='1 Session'),
                Patch(facecolor='#FFD166', edgecolor='black', label='2 Sessions'),
                Patch(facecolor='#FF6B6B', edgecolor='black', label='3 Sessions'),
                Patch(facecolor='white', edgecolor='black', label='Error bars: Confidence Interval')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            # Confidence interval values लिखो (selected points पर)
            for i in range(0, len(df), 8):  # Every 8th bar
                bar = bars[i]
                height = bar.get_height()
                lower = df['confidence_lower'].iloc[i]
                upper = df['confidence_upper'].iloc[i]
                
                if height > 0:  # Only for non-zero predictions
                    ax.text(bar.get_x() + bar.get_width()/2., upper + 0.1,
                        f'{lower}-{upper}', 
                        ha='center', va='bottom', fontsize=8,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
            plt.tight_layout()
            plt.savefig(f"{self.PLOTS_PATH}/7day_forecast_with_ci_{model_name}.png", dpi=300)
            plt.show()
            
            # Confidence interval statistics print करो
            print(f"\n📊 Confidence Interval Statistics:")
            print(f"   Average CI width: {(df['confidence_upper'] - df['confidence_lower']).mean():.2f}")
            print(f"   Min CI width: {(df['confidence_upper'] - df['confidence_lower']).min():.2f}")
            print(f"   Max CI width: {(df['confidence_upper'] - df['confidence_lower']).max():.2f}")
        
            def plot_price_forecast(self, backend_json, model_name="Model"):
                """Price forecast प्लॉट (backend JSON से)"""
                if not backend_json or 'hourly_forecasts' not in backend_json:
                    print("❌ No price data in backend JSON")
                    return
                
                forecasts = backend_json['hourly_forecasts']
                df = pd.DataFrame(forecasts)
                
                # टाइमस्टैम्प प्रोसेस करें
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Single plot for price forecast
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))
                
                # 1. PRICE PLOT (top)
                # Colors based on price level
                price_colors = []
                for _, row in df.iterrows():
                    demand_level = row['pricing']['demand_level']
                    if demand_level == 'HIGH':
                        price_colors.append('#FF6B6B')  # Red
                    elif demand_level == 'MEDIUM':
                        price_colors.append('#FFD166')  # Yellow
                    else:  # LOW
                        price_colors.append('#4ECDC4')  # Teal
                
                # बार्स for price
                x_pos = np.arange(len(df))
                price_bars = ax1.bar(x_pos, 
                                    df['pricing'].apply(lambda x: x['suggested_price_per_kwh']),
                                    color=price_colors,
                                    edgecolor='black',
                                    alpha=0.8)
                
                # X-axis labels (every 12 hours)
                tick_positions = []
                tick_labels = []
                for i in range(0, len(df), 12):
                    tick_positions.append(i)
                    tick_labels.append(df['timestamp'].iloc[i].strftime('%d %b\n%H:00'))
                
                ax1.set_xticks(tick_positions)
                ax1.set_xticklabels(tick_labels, fontsize=10, rotation=45, ha='right')
                
                # Y-axis for price
                ax1.set_ylabel('Price (₹/kWh)', fontsize=12)
                ax1.set_ylim(6, 15)  # Fixed range for price
                
                # Title
                ax1.set_title(f'7-Day Dynamic Pricing Forecast\n({model_name})', 
                            fontsize=16, fontweight='bold')
                
                # Grid
                ax1.grid(True, alpha=0.3, axis='y')
                
                # Price values लिखो
                for bar, price in zip(price_bars, df['pricing'].apply(lambda x: x['suggested_price_per_kwh'])):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'₹{price:.1f}', 
                            ha='center', va='bottom', fontsize=8,
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                
                # Price level legend
                from matplotlib.patches import Patch
                price_legend = [
                    Patch(facecolor='#4ECDC4', edgecolor='black', label='LOW Demand (₹8.0/kWh)'),
                    Patch(facecolor='#FFD166', edgecolor='black', label='MEDIUM Demand (₹10.0/kWh)'),
                    Patch(facecolor='#FF6B6B', edgecolor='black', label='HIGH Demand (₹13.0/kWh)')
                ]
                ax1.legend(handles=price_legend, loc='upper right', fontsize=10)
                
                # 2. DEMAND vs PRICE COMPARISON (bottom)
                # Get demand data if available
                if 'predicted_demand' in df.columns:
                    # Dual axis plot
                    ax2_demand = ax2
                    ax2_price = ax2.twinx()
                    
                    # Demand line
                    demand_line = ax2_demand.plot(df['timestamp'], 
                                                df['predicted_demand'],
                                                color='#2E86AB',
                                                linewidth=2,
                                                marker='o',
                                                markersize=3,
                                                label='Demand (Sessions)')[0]
                    
                    # Price line
                    price_line = ax2_price.plot(df['timestamp'],
                                            df['pricing'].apply(lambda x: x['suggested_price_per_kwh']),
                                            color='#FF6B6B',
                                            linewidth=2,
                                            linestyle='--',
                                            marker='s',
                                            markersize=3,
                                            label='Price (₹/kWh)')[0]
                    
                    # Labels
                    ax2_demand.set_xlabel('Date & Time', fontsize=12)
                    ax2_demand.set_ylabel('Demand (Sessions)', fontsize=12, color='#2E86AB')
                    ax2_price.set_ylabel('Price (₹/kWh)', fontsize=12, color='#FF6B6B')
                    
                    # Title
                    ax2_demand.set_title('Demand vs Price Correlation', fontsize=14, fontweight='bold')
                    
                    # Grid
                    ax2_demand.grid(True, alpha=0.3)
                    
                    # X-ticks format
                    ax2_demand.xaxis.set_major_formatter(DateFormatter('%d %b\n%H:00'))
                    plt.setp(ax2_demand.xaxis.get_majorticklabels(), rotation=45, ha='right')
                    
                    # Legends
                    lines = [demand_line, price_line]
                    labels = [l.get_label() for l in lines]
                    ax2_demand.legend(lines, labels, loc='upper left')
                    
                    # Correlation coefficient
                    try:
                        correlation = np.corrcoef(df['predicted_demand'], 
                                                df['pricing'].apply(lambda x: x['suggested_price_per_kwh']))[0, 1]
                        ax2_demand.text(0.02, 0.98, f'Correlation: {correlation:.3f}',
                                    transform=ax2_demand.transAxes,
                                    fontsize=10,
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    except:
                        pass
                
                plt.tight_layout()
                plt.savefig(f"{self.PLOTS_PATH}/price_forecast_{model_name}.png", dpi=300)
                plt.show()
                
                # Price statistics print करो
                print(f"\n💰 Price Forecast Statistics:")
                prices = df['pricing'].apply(lambda x: x['suggested_price_per_kwh'])
                print(f"   Average Price: ₹{prices.mean():.2f}/kWh")
                print(f"   Min Price: ₹{prices.min():.2f}/kWh")
                print(f"   Max Price: ₹{prices.max():.2f}/kWh")
                
                # Demand level distribution
                demand_levels = df['pricing'].apply(lambda x: x['demand_level'])
                print(f"\n📊 Demand Level Distribution:")
                for level in ['LOW', 'MEDIUM', 'HIGH']:
                    count = (demand_levels == level).sum()
                    percentage = (count / len(demand_levels)) * 100
                    print(f"   {level}: {count} hours ({percentage:.1f}%)")



    def plot_price_forecast_from_backend(self, backend_json_file, model_name="Model"):
        """Backend JSON file से price forecast प्लॉट"""
        try:
            # JSON file load करो
            with open(backend_json_file, 'r') as f:
                backend_json = json.load(f)
            
            print(f"✅ Loaded backend JSON from: {backend_json_file}")
            return self.plot_price_forecast(backend_json, model_name)
            
        except Exception as e:
            print(f"❌ Error loading backend JSON: {e}")
            return None
    
    def plot_price_forecast(self, backend_json, model_name="Model"):
        """Price forecast प्लॉट - backend JSON से"""
        if not backend_json or 'hourly_forecasts' not in backend_json:
            print("❌ No hourly_forecasts in backend JSON")
            return
        
        forecasts = backend_json['hourly_forecasts']
        
        # DataFrame बनाओ
        data = []
        for fc in forecasts:
            data.append({
                'timestamp': fc['timestamp'],
                'predicted_demand': fc['predicted_demand'],
                'demand_level': fc['pricing']['demand_level'],
                'price': fc['pricing']['suggested_price_per_kwh'],
                'price_multiplier': fc['pricing']['price_multiplier'],
                'confidence_lower': fc['confidence_interval']['lower'],
                'confidence_upper': fc['confidence_interval']['upper']
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(18, 15))
        
        # 1. PRICE BAR CHART (top)
        ax1 = axes[0]
        
        # Colors based on demand level
        price_colors = df['demand_level'].map({
            'LOW': '#4ECDC4',    # Teal
            'MEDIUM': '#FFD166',  # Yellow
            'HIGH': '#FF6B6B'     # Red
        })
        
        # बार्स for price
        x_pos = np.arange(len(df))
        price_bars = ax1.bar(x_pos, 
                            df['price'],
                            color=price_colors,
                            edgecolor='black',
                            alpha=0.8,
                            width=0.8)
        
        # X-axis labels (every 12 hours)
        tick_positions = []
        tick_labels = []
        for i in range(0, len(df), 12):
            tick_positions.append(i)
            tick_labels.append(df['timestamp'].iloc[i].strftime('%d %b\n%H:00'))
        
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels(tick_labels, fontsize=10, rotation=45, ha='right')
        
        # Y-axis for price
        ax1.set_ylabel('Price (₹/kWh)', fontsize=12)
        ax1.set_ylim(6, 16)  # Fixed range for price
        ax1.set_yticks([8, 10, 13, 15])
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Title
        ax1.set_title(f'7-Day Dynamic Pricing Forecast\n({model_name})', 
                     fontsize=16, fontweight='bold')
        
        # Price values लिखो (selected bars पर)
        for i in range(0, len(df), 8):  # Every 8th bar
            bar = price_bars[i]
            price = df['price'].iloc[i]
            ax1.text(bar.get_x() + bar.get_width()/2., price + 0.2,
                    f'₹{price:.1f}', 
                    ha='center', va='bottom', fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. DEMAND vs PRICE COMPARISON (middle)
        ax2 = axes[1]
        
        # Demand bars
        demand_colors = df['predicted_demand'].map({
            0: '#95E1D3',  # Very light teal
            1: '#4ECDC4',  # Teal
            2: '#FFD166',  # Yellow
            3: '#FF6B6B'   # Red
        })
        
        demand_bars = ax2.bar(x_pos, 
                             df['predicted_demand'],
                             color=demand_colors,
                             edgecolor='black',
                             alpha=0.8,
                             width=0.8)
        
        # Confidence intervals as error bars
        ax2.errorbar(x_pos, df['predicted_demand'],
                    yerr=[df['predicted_demand'] - df['confidence_lower'],
                          df['confidence_upper'] - df['predicted_demand']],
                    fmt='none', ecolor='black', capsize=3, alpha=0.5)
        
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels(tick_labels, fontsize=10, rotation=45, ha='right')
        
        # Y-axis for demand
        ax2.set_ylabel('Demand (Sessions)', fontsize=12)
        ax2.set_ylim(0, 4)
        ax2.set_yticks([0, 1, 2, 3])
        ax2.grid(True, alpha=0.3, axis='y')
        
        ax2.set_title('Demand Forecast with Confidence Intervals', 
                     fontsize=14, fontweight='bold')
        
        # Demand values लिखो
        for i in range(0, len(df), 8):
            bar = demand_bars[i]
            demand = df['predicted_demand'].iloc[i]
            lower = df['confidence_lower'].iloc[i]
            upper = df['confidence_upper'].iloc[i]
            
            ax2.text(bar.get_x() + bar.get_width()/2., demand + 0.1,
                    f'{demand} [{lower}-{upper}]', 
                    ha='center', va='bottom', fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # 3. PRICE MULTIPLIER TIMELINE (bottom)
        ax3 = axes[2]
        
        # Line for price multiplier
        ax3.plot(df['timestamp'], df['price_multiplier'],
                color='#2E86AB',
                linewidth=2.5,
                marker='o',
                markersize=4,
                label='Price Multiplier')
        
        # Fill for different demand levels
        for level, color in [('LOW', '#4ECDC4'), ('MEDIUM', '#FFD166'), ('HIGH', '#FF6B6B')]:
            level_mask = df['demand_level'] == level
            if level_mask.any():
                ax3.fill_between(df['timestamp'][level_mask], 0.7, 1.4,
                                alpha=0.2, color=color, label=f'{level} Demand')
        
        # X-axis format
        ax3.xaxis.set_major_formatter(DateFormatter('%d %b\n%H:00'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Y-axis
        ax3.set_ylabel('Price Multiplier', fontsize=12)
        ax3.set_ylim(0.7, 1.4)
        ax3.set_yticks([0.8, 1.0, 1.3])
        ax3.set_yticklabels(['0.8x (LOW)', '1.0x (MED)', '1.3x (HIGH)'])
        ax3.grid(True, alpha=0.3)
        
        ax3.set_title('Dynamic Pricing Multiplier Over Time', 
                     fontsize=14, fontweight='bold')
        
        # Legend
        ax3.legend(loc='upper right')
        
        # Horizontal lines for reference
        ax3.axhline(y=0.8, color='#4ECDC4', linestyle='--', alpha=0.5)
        ax3.axhline(y=1.0, color='#FFD166', linestyle='--', alpha=0.5)
        ax3.axhline(y=1.3, color='#FF6B6B', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(f"{self.PLOTS_PATH}/price_forecast_detailed_{model_name}.png", dpi=300)
        plt.show()
        
        # Statistics print करो
        self._print_price_statistics(df)
    
    def _print_price_statistics(self, df):
        """Price statistics print करो"""
        print("\n" + "="*70)
        print("💰 PRICE FORECAST STATISTICS")
        print("="*70)
        
        print(f"\n📊 Price Summary:")
        print(f"   Average Price: ₹{df['price'].mean():.2f}/kWh")
        print(f"   Minimum Price: ₹{df['price'].min():.2f}/kWh")
        print(f"   Maximum Price: ₹{df['price'].max():.2f}/kWh")
        print(f"   Standard Deviation: ₹{df['price'].std():.2f}/kWh")
        
        print(f"\n📊 Demand Level Distribution:")
        demand_counts = df['demand_level'].value_counts()
        for level, count in demand_counts.items():
            percentage = (count / len(df)) * 100
            price = df[df['demand_level'] == level]['price'].mean()
            print(f"   {level}: {count} hours ({percentage:.1f}%) - Avg: ₹{price:.1f}/kWh")
        
        print(f"\n📊 Session Distribution:")
        session_counts = df['predicted_demand'].value_counts().sort_index()
        for sessions, count in session_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {sessions} sessions: {count} hours ({percentage:.1f}%)")
        
        print(f"\n📈 Confidence Intervals:")
        print(f"   Average CI width: {(df['confidence_upper'] - df['confidence_lower']).mean():.2f}")
        print(f"   Narrowest CI: {(df['confidence_upper'] - df['confidence_lower']).min():.2f}")
        print(f"   Widest CI: {(df['confidence_upper'] - df['confidence_lower']).max():.2f}")
        
        print(f"\n💡 Best Times to Charge (Lowest Price):")
        low_price_times = df.nsmallest(5, 'price')[['timestamp', 'price', 'predicted_demand']]
        for idx, row in low_price_times.iterrows():
            time_str = pd.to_datetime(row['timestamp']).strftime('%a %d %b %H:00')
            print(f"   {time_str}: ₹{row['price']:.1f}/kWh, {row['predicted_demand']} sessions")