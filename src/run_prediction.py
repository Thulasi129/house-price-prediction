
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import os

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# --- 1. Load Data ---
df = pd.read_csv('data/house_prices.csv')
print("--- Data Loaded ---")
print(df.head())

# --- 2. Exploratory Data Analysis (EDA) ---
print("\n--- Starting EDA ---")

# Save summary statistics
with open('results/summary_stats.txt', 'w') as f:
    f.write("Info:\n")
    df.info(buf=f)
    f.write("\n\nDescribe:\n")
    f.write(str(df.describe()))

# Visualizations
sns.set_style('whitegrid')

# Distribution plots
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
sns.histplot(df['area'], kde=True, bins=30)
plt.title('Distribution of Area')
plt.subplot(2, 2, 2)
sns.histplot(df['bedrooms'], kde=False, bins=5)
plt.title('Distribution of Bedrooms')
plt.subplot(2, 2, 3)
sns.histplot(df['age'], kde=True, bins=30)
plt.title('Distribution of Age')
plt.subplot(2, 2, 4)
sns.histplot(df['price'], kde=True, bins=30)
plt.title('Distribution of Price')
plt.tight_layout()
plt.savefig('results/distributions.png')
plt.close()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='location', y='price', data=df)
plt.title('Price Distribution by Location')
plt.savefig('results/price_by_location.png')
plt.close()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr(numeric_only=True)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig('results/correlation_matrix.png')
plt.close()

# Pair plot
sns.pairplot(df, x_vars=['area', 'bedrooms', 'age'], y_vars=['price'], height=5, aspect=0.8, kind='scatter')
plt.savefig('results/pairplot.png')
plt.close()

print("EDA visualizations saved to 'results' directory.")

# --- 3. Data Preprocessing ---
print("\n--- Preprocessing Data ---")
df_processed = pd.get_dummies(df, columns=['location'], drop_first=True, dtype=int)

X = df_processed.drop('price', axis=1)
y = df_processed['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
numerical_features = ['area', 'bedrooms', 'age']
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
print("Data preprocessing complete.")

# --- 4. Model Implementation ---
print("\n--- Training Models ---")
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'R-squared': r2, 'model': model}
print("Model training complete.")

# --- 5. Model Evaluation ---
print("\n--- Evaluating Models ---")
results_df = pd.DataFrame({(i): results[i] for i in results.keys()}).T[['RMSE', 'R-squared']]
print("Model Performance:")
print(results_df)
results_df.to_csv('results/model_performance.csv')

# Bar plot of RMSE
plt.figure(figsize=(10, 6))
sns.barplot(x=results_df.index, y=results_df['RMSE'])
plt.title('Model Comparison: RMSE')
plt.ylabel('Root Mean Squared Error')
plt.savefig('results/model_rmse_comparison.png')
plt.close()

# Actual vs. Predicted plots
plt.figure(figsize=(18, 5))
for i, (name, result) in enumerate(results.items()):
    plt.subplot(1, 3, i + 1)
    y_pred = result['model'].predict(X_test)
    sns.scatterplot(x=y_test, y=y_pred)
    plt.title(name)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.tight_layout()
plt.savefig('results/actual_vs_predicted.png')
plt.close()

# Feature Importance
feature_importance = pd.DataFrame()
for name, result in results.items():
    if hasattr(result['model'], 'coef_'):
        feature_importance[name] = result['model'].coef_
feature_importance.index = X_train.columns
print("\nFeature Importance:")
print(feature_importance)
feature_importance.to_csv('results/feature_importance.csv')

# Feature importance plot
feature_importance.plot(kind='bar', figsize=(15, 8))
plt.title('Feature Importance across Models')
plt.ylabel('Coefficient Value')
plt.savefig('results/feature_importance.png')
plt.close()

print("\nModel evaluation complete. All results saved to 'results' directory.")

# --- 6. Save Model, Scaler, and Columns for Streamlit App ---
import joblib
joblib.dump(models['Ridge Regression'], 'results/model.joblib')
joblib.dump(scaler, 'results/scaler.joblib')
joblib.dump(X_train.columns.tolist(), 'results/model_columns.joblib')

print("\nModel, scaler, and columns saved for Streamlit app.")
