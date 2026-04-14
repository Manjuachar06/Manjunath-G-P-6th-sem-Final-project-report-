"""
Resume Classification & Scoring Dataset
Dataset from: Resume Classification Task (Kaggle-style)
Predicts: Resume quality/match score (0-100)

This script creates a resume classification dataset with features extracted from resume text,
then trains two ML models (Logistic Regression and Random Forest) for resume scoring.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge  # Regression model
from sklearn.ensemble import RandomForestRegressor  # Regression model
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import os

# Create output directories
os.makedirs('ml_results', exist_ok=True)

print("="*70)
print("RESUME SCORING DATASET & MODEL TRAINING")
print("="*70)

# ============================================================================
# STEP 1: Create/Load Resume Dataset
# ============================================================================
print("\n[1/4] Creating Resume Dataset...")

# Dataset Features (extracted resume metrics)
np.random.seed(42)
n_samples = 500

data = {
    # Text complexity features
    'word_count': np.random.uniform(100, 800, n_samples),  # Resume length
    'avg_sentence_length': np.random.uniform(5, 25, n_samples),
    'keyword_relevance_score': np.random.uniform(0.1, 0.95, n_samples),  # Keywords match to job
    
    # Education & Experience
    'education_level': np.random.choice([1, 2, 3, 4], n_samples),  # 1=HS, 2=Bachelor, 3=Master, 4=PhD
    'years_experience': np.random.uniform(0, 30, n_samples),
    'skills_count': np.random.randint(1, 25, n_samples),
    'certification_count': np.random.randint(0, 10, n_samples),
    'project_count': np.random.randint(0, 20, n_samples),
    
    # Grammar & Format
    'grammar_score': np.random.uniform(0.4, 1.0, n_samples),  # 0-1 scale
    'formatting_score': np.random.uniform(0.3, 1.0, n_samples),  # 0-1 scale
    'has_gpa': np.random.choice([0, 1], n_samples),
    'has_linkedin': np.random.choice([0, 1], n_samples),
    'has_portfolio': np.random.choice([0, 1], n_samples),
}

df = pd.DataFrame(data)

# TARGET: Resume Quality/Match Score (0-100)
# Create target based on features (simulate real resume scoring)
target = (
    (df['keyword_relevance_score'] * 0.3) +  # 30% keyword relevance
    (df['education_level'] * 0.15 / 4) +      # 15% education
    (np.minimum(df['years_experience'] / 30, 1) * 0.2) +  # 20% experience
    (np.minimum(df['skills_count'] / 25, 1) * 0.15) +     # 15% skills
    (df['grammar_score'] * 0.1) +             # 10% grammar
    (df['formatting_score'] * 0.05) +         # 5% formatting
    (df['has_portfolio'] * 0.05)              # 5% portfolio
) * 100

# Add some realistic noise
target = target + np.random.normal(0, 5, n_samples)
target = np.clip(target, 0, 100)  # Keep in 0-100 range

df['resume_score'] = target

# Save dataset
df.to_csv('ml_results/resume_dataset.csv', index=False)
print(f"✓ Dataset created: {len(df)} resume samples")
print(f"  Features: {df.shape[1] - 1} (11 features + 1 target)")
print(f"  Target (resume_score): min={target.min():.2f}, max={target.max():.2f}, mean={target.mean():.2f}")

# ============================================================================
# STEP 2: Prepare Data for Training
# ============================================================================
print("\n[2/4] Preparing Data...")

X = df.drop('resume_score', axis=1)
y = df['resume_score']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Test set: {X_test.shape[0]} samples")

# Save scaler for later use
joblib.dump(scaler, 'ml_results/scaler.pkl')

# ============================================================================
# STEP 3: Train Two Models
# ============================================================================
print("\n[3/4] Training Models...")

models = {
    'Ridge Regression': Ridge(alpha=1.0, random_state=42),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=15)
}

results = {}

for model_name, model in models.items():
    print(f"\n  Training {model_name}...")
    
    # Train
    if model_name == 'Ridge Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    results[model_name] = {
        'model': model,
        'y_pred': y_pred,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'scaler': scaler if model_name == 'Ridge Regression' else None
    }
    
    print(f"    • R² Score:   {r2:.4f}")
    print(f"    • RMSE:       {rmse:.4f}")
    print(f"    • MAE:        {mae:.4f}")
    print(f"    • MAPE:       {mape:.2f}%")
    
    # Save model
    joblib.dump(model, f'ml_results/{model_name.lower().replace(" ", "_")}_model.pkl')

# ============================================================================
# STEP 4: Create Visualizations & Metrics
# ============================================================================
print("\n[4/4] Generating Visualizations...")

# Save metrics comparison
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R² Score': [results[m]['r2'] for m in results.keys()],
    'RMSE': [results[m]['rmse'] for m in results.keys()],
    'MAE': [results[m]['mae'] for m in results.keys()],
    'MAPE (%)': [results[m]['mape'] for m in results.keys()]
})

print("\n" + "="*70)
print("MODEL COMPARISON SUMMARY")
print("="*70)
print(comparison_df.to_string(index=False))
comparison_df.to_csv('ml_results/model_comparison.csv', index=False)

plt.style.use('seaborn-v0_8-darkgrid')

# 1. Model Performance Comparison
fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['R² Score', 'RMSE', 'MAE', 'MAPE (%)']
x = np.arange(len(metrics))
width = 0.35

for i, model_name in enumerate(results.keys()):
    values = [
        results[model_name]['r2'],
        results[model_name]['rmse'],
        results[model_name]['mae'],
        results[model_name]['mape']
    ]
    ax.bar(x + i*width, values, width, label=model_name, alpha=0.85, edgecolor='black')

ax.set_xlabel('Metrics', fontsize=13, fontweight='bold')
ax.set_ylabel('Score Value', fontsize=13, fontweight='bold')
ax.set_title('Resume Scoring Models - Performance Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x + width/2)
ax.set_xticklabels(metrics)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ml_results/01_model_performance_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 01_model_performance_comparison.png")
plt.close()

# 2. Actual vs Predicted (Scatter plots)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (model_name, result) in enumerate(results.items()):
    axes[idx].scatter(y_test, result['y_pred'], alpha=0.6, edgecolors='k', s=50)
    axes[idx].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[idx].set_xlabel('Actual Resume Score', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Predicted Resume Score', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{model_name}\n(R² = {result["r2"]:.4f})', fontweight='bold', fontsize=12)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_results/02_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 02_actual_vs_predicted.png")
plt.close()

# 3. Residuals Analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (model_name, result) in enumerate(results.items()):
    residuals = y_test.values - result['y_pred']
    axes[idx].scatter(result['y_pred'], residuals, alpha=0.6, edgecolors='k', s=50)
    axes[idx].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[idx].set_xlabel('Predicted Resume Score', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Residuals', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{model_name} - Residual Plot', fontweight='bold', fontsize=12)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_results/03_residuals_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 03_residuals_analysis.png")
plt.close()

# 4. Feature Importance (Random Forest)
if 'Random Forest Regressor' in results:
    model = results['Random Forest Regressor']['model']
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]  # Top 10 features
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_importances = importances[indices]
    feature_names = X.columns[indices]
    
    bars = ax.barh(range(len(indices)), top_importances, color='#8b5cf6', alpha=0.85, edgecolor='black')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Importance Score', fontsize=13, fontweight='bold')
    ax.set_title('Random Forest - Top 10 Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, imp) in enumerate(zip(bars, top_importances)):
        ax.text(imp + 0.003, i, f'{imp:.4f}', va='center', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('ml_results/06_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: 06_feature_importance.png")
    plt.close()

# 5. Prediction Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for idx, (model_name, result) in enumerate(results.items()):
    axes[idx].hist(y_test.values, bins=20, alpha=0.5, label='Actual', edgecolor='black')
    axes[idx].hist(result['y_pred'], bins=20, alpha=0.5, label='Predicted', edgecolor='black')
    axes[idx].set_xlabel('Resume Score', fontsize=11, fontweight='bold')
    axes[idx].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[idx].set_title(f'{model_name} - Score Distribution', fontweight='bold', fontsize=12)
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_results/07_score_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: 07_score_distribution.png")
plt.close()

# Print detailed metrics
print("\n" + "="*70)
print("DETAILED MODEL METRICS")
print("="*70)

for model_name, result in results.items():
    print(f"\n{model_name.upper()}")
    print("-" * 70)
    print(f"  R² Score:                    {result['r2']:.6f}")
    print(f"  Root Mean Squared Error:     {result['rmse']:.6f}")
    print(f"  Mean Absolute Error:         {result['mae']:.6f}")
    print(f"  Mean Absolute Percentage:    {result['mape']:.4f}%")
    print(f"  Mean Squared Error:          {result['mse']:.6f}")

print("\n" + "="*70)
print("✓ TRAINING COMPLETE!")
print("="*70)
print("\nGenerated Files in 'ml_results/':")
print("  📊 Datasets:")
print("     • resume_dataset.csv - Training dataset (500 samples)")
print("     • model_comparison.csv - Performance metrics")
print("  🤖 Models:")
print("     • ridge_regression_model.pkl - Ridge Regression model")
print("     • random_forest_regressor_model.pkl - Random Forest model")
print("     • scaler.pkl - Feature scaler for Ridge model")
print("  📈 Visualizations:")
print("     • 01_model_performance_comparison.png")
print("     • 02_actual_vs_predicted.png")
print("     • 03_residuals_analysis.png")
print("     • 06_feature_importance.png")
print("     • 07_score_distribution.png")
print("\n✅ Ready to integrate with Flask app!")
