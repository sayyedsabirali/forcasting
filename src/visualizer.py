import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from config import PLOTS_PATH

class Visualizer:
    
    def __init__(self):
        os.makedirs(PLOTS_PATH, exist_ok=True)
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
    
    def plot_feature_importance(self, models, X_train, feature_names=None):
            """फीचर इंपोर्टेंस प्लॉट करें"""
            
            # फीचर नेम्स हासिल करें
            if hasattr(X_train, 'columns'):
                # X_train pandas DataFrame है
                feature_names = list(X_train.columns)
            elif feature_names is not None:
                # बाहर से नेम्स मिले
                feature_names = feature_names
            else:
                # डिफॉल्ट नेम्स
                if hasattr(X_train, 'shape'):
                    feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
                else:
                    feature_names = [f'Feature_{i}' for i in range(len(X_train[0]))]
            
            model_names = list(models.keys())
            n_models = len(model_names)
            
            fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 8))
            
            if n_models == 1:
                axes = [axes]
            
            for idx, model_name in enumerate(model_names):
                model = models[model_name]
                
                try:
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                    else:
                        print(f"{model_name} - No feature importances")
                        axes[idx].text(0.5, 0.5, "Not available", 
                                    ha='center', va='center')
                        axes[idx].set_title(model_name)
                        continue
                    
                    # टॉप 10 फीचर्स
                    indices = np.argsort(importances)[-10:]
                    
                    axes[idx].barh(range(len(indices)), importances[indices])
                    axes[idx].set_yticks(range(len(indices)))
                    
                    # फीचर नेम्स (पहले 20 characters)
                    names = []
                    for i in indices:
                        name = feature_names[i]
                        if len(name) > 20:
                            name = name[:17] + "..."
                        names.append(name)
                    
                    axes[idx].set_yticklabels(names)
                    axes[idx].set_xlabel('Importance')
                    axes[idx].set_title(f'{model_name} - Top Features')
                    axes[idx].invert_yaxis()
                    
                except Exception as e:
                    print(f"Error in {model_name}: {e}")
                    axes[idx].text(0.5, 0.5, f"Error\n{e}", 
                                ha='center', va='center')
                    axes[idx].set_title(model_name)
            
            plt.tight_layout()
            plt.savefig(f"{PLOTS_PATH}/feature_importance.png")
            plt.show()
            
            plt.tight_layout()
            plt.savefig(f"{PLOTS_PATH}/feature_importance_all.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # अलग-अलग फाइल्स भी सेव करें
            for idx, model_name in enumerate(model_names):
                try:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    model = models[model_name]
                    
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        indices = np.argsort(importances)[::-1][:20]
                        
                        colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(indices)))
                        ax.barh(range(len(indices)), importances[indices], 
                            color=colors, edgecolor='black')
                        
                        ax.set_yticks(range(len(indices)))
                        
                        # फीचर नेम्स फॉर्मेट करें
                        display_names = []
                        for i in indices:
                            name = feature_names[i]
                            # रीडेबल नेम्स
                            name = name.replace('_', ' ').title()
                            name = name.replace('Lag', 'Lag ')
                            name = name.replace('Rolling', 'Rolling ')
                            display_names.append(name)
                        
                        ax.set_yticklabels(display_names, fontsize=10)
                        ax.set_xlabel('Importance Score', fontsize=12)
                        ax.set_title(f'Feature Importance - {model_name}', 
                                fontsize=14, fontweight='bold')
                        ax.invert_yaxis()
                        ax.grid(True, alpha=0.3, axis='x')
                        
                        plt.tight_layout()
                        plt.savefig(f"{PLOTS_PATH}/feature_importance_{model_name}.png", 
                                dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        
                except Exception as e:
                    print(f"  Skipping individual plot for {model_name}: {e}")
                    plt.close(fig)
    
    def plot_model_comparison(self, results):
        """मॉडल्स का कम्पेरिजन प्लॉट"""
        model_names = list(results.keys())
        
        if not model_names:
            print("⚠️ No models to compare")
            return
        
        # डेटा तैयार करें
        metrics_data = {
            'MAE': [results[name]['mae'] for name in model_names],
            'RMSE': [results[name]['rmse'] for name in model_names],
            'R² Score': [results[name]['r2'] for name in model_names]
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # अच्छे रंग
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            ax = axes[idx]
            
            # बार प्लॉट
            bars = ax.bar(model_names, values, color=colors[idx], 
                         edgecolor='black', linewidth=1.5, alpha=0.8)
            
            ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
            ax.set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
            
            # X-टिक लेबल्स
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=11)
            
            # वैल्यूज लिखें
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10,
                       fontweight='bold')
            
            # R² के लिए विशेष फॉर्मेट
            if metric_name == 'R² Score':
                ax.set_ylim([0, 1])
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
                ax.text(0.02, 0.95, 'Higher is better', 
                       transform=ax.transAxes, fontsize=10, 
                       style='italic', alpha=0.7)
            else:
                ax.text(0.02, 0.95, 'Lower is better', 
                       transform=ax.transAxes, fontsize=10, 
                       style='italic', alpha=0.7)
            
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_PATH}/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # टेबल भी बनाएं
        print("\n" + "="*60)
        print("📊 MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        from tabulate import tabulate
        table_data = []
        for name in model_names:
            table_data.append([
                name,
                f"{results[name]['mae']:.3f}",
                f"{results[name]['rmse']:.3f}",
                f"{results[name]['r2']:.3f}"
            ])
        
        print(tabulate(table_data, 
                      headers=['Model', 'MAE', 'RMSE', 'R² Score'],
                      tablefmt='grid'))
    
    def plot_actual_vs_predicted(self, y_test, predictions_dict, model_names, feature_names=None):
        """सभी मॉडल्स के actual vs predicted"""
        n_models = len(model_names)
        
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, model_name in enumerate(model_names):
            y_pred = predictions_dict[model_name]
            
            ax = axes[idx]
            
            # स्कैटर प्लॉट
            scatter = ax.scatter(y_test, y_pred, 
                               alpha=0.6, 
                               s=50,
                               c=y_test,  # कलर कोडिंग
                               cmap='viridis',
                               edgecolor='black',
                               linewidth=0.5)
            
            # परफेक्ट प्रेडिक्शन लाइन
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([0, max_val], [0, max_val], 
                   'r--', linewidth=2, alpha=0.7, label='Perfect Prediction')
            
            # बेस्ट फिट लाइन
            try:
                z = np.polyfit(y_test, y_pred, 1)
                p = np.poly1d(z)
                ax.plot(y_test, p(y_test), 
                       'g-', linewidth=2, alpha=0.7, label='Best Fit')
            except:
                pass
            
            ax.set_xlabel('Actual Sessions', fontsize=12)
            ax.set_ylabel('Predicted Sessions', fontsize=12)
            ax.set_title(f'{model_name}', fontsize=14, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # R² वैल्यू डालें
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test, y_pred)
            ax.text(0.05, 0.95, f'R² = {r2:.3f}', 
                   transform=ax.transAxes, fontsize=11,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.suptitle('Actual vs Predicted Charging Sessions', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f"{PLOTS_PATH}/actual_vs_predicted_all.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_forecast(self, forecast_results, model_name="Model"):
        """7-दिन के फोरकास्ट को प्लॉट करें"""
        df = pd.DataFrame(forecast_results)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure integer
        df['predicted_sessions'] = df['predicted_sessions'].astype(int)
        
        print(f"\n📊 Forecast Distribution for {model_name}:")
        for val in range(4):
            count = (df['predicted_sessions'] == val).sum()
            print(f"   {val}: {count} hours")

        
        # 2 सबप्लॉट: लाइन प्लॉट और हीटमैप
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))
        
        # 1. लाइन प्लॉट
        ax1 = axes[0]
        
        # फोरकास्ट लाइन
        line = ax1.plot(df['timestamp'], df['predicted_sessions'], 
                    marker='o', markersize=4, linewidth=2, 
                    label='Forecast', color='#2E86AB')
        
        # कॉन्फिडेंस इंटरवल
        if 'confidence_lower' in df.columns and 'confidence_upper' in df.columns:
            ax1.fill_between(df['timestamp'], 
                        df['confidence_lower'], 
                        df['confidence_upper'],
                        alpha=0.3, label='80% Confidence', color='#2E86AB')
        
        ax1.set_xlabel('Date & Time', fontsize=12)
        ax1.set_ylabel('Predicted Sessions', fontsize=12)
        ax1.set_title(f'7-Day EV Charging Demand Forecast - {model_name}', 
                    fontsize=16, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # X-टिक्स को अच्छे फॉर्मेट में
        from matplotlib.dates import DateFormatter, HourLocator
        ax1.xaxis.set_major_formatter(DateFormatter('%d %b\n%H:00'))
        
        # Y-axis को integer में
        ax1.set_yticks([0, 1, 2, 3])
        
        # 2. हीटमैप
        ax2 = axes[1]
        
        # हीटमैप डेटा तैयार करें
        df['hour'] = df['timestamp'].dt.hour
        df['day'] = df['timestamp'].dt.date
        df['day_name'] = df['timestamp'].dt.day_name()
        
        # पिवट टेबल
        pivot_df = df.pivot_table(values='predicted_sessions', 
                                index='day_name', 
                                columns='hour', 
                                aggfunc='mean')
        
        # Round to integer
        pivot_df = pivot_df.round(0).astype(int)
        
        # डेज के क्रम में
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                    'Friday', 'Saturday', 'Sunday']
        
        # Reindex और fill missing days with 0
        pivot_df = pivot_df.reindex(days_order, fill_value=0)
        
        # Fill missing hours with 0
        for h in range(24):
            if h not in pivot_df.columns:
                pivot_df[h] = 0
        
        pivot_df = pivot_df.sort_index(axis=1)
        
        # हीटमैप
        import seaborn as sns
        sns.heatmap(pivot_df, 
                ax=ax2,
                cmap='YlOrRd',
                annot=True,
                fmt='d',  # Integer format
                cbar_kws={'label': 'Sessions', 'shrink': 0.8},
                linewidths=0.5,
                linecolor='gray',
                vmin=0,
                vmax=3)
        
        ax2.set_title('Weekly Demand Heatmap (Hourly)', 
                    fontsize=16, fontweight='bold')
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Day of Week', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f"{PLOTS_PATH}/7day_forecast_{model_name}.png", 
                dpi=300, bbox_inches='tight')
        plt.show()
        
        # हीटमैप डेटा सेव करें
        pivot_df.to_csv(f"{PLOTS_PATH}/heatmap_data_{model_name}.csv")
        print(f"✅ Heatmap data saved to: {PLOTS_PATH}/heatmap_data_{model_name}.csv")