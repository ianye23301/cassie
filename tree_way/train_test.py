import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load data (Modify this path)
data = pd.read_csv("../data.csv")  # Replace with your actual dataset
target_column = "price"  # Replace with your actual target column

# Identify categorical columns
categorical_columns = ["transmission", "model", "fuelType"]  # Modify this list based on your dataset

# One-Hot Encoding
encoder = joblib.load("encoder.pkl")  # Load the encoder
X_encoded = encoder.transform(data[categorical_columns])  # Transform categorical data
encoded_feature_names = encoder.get_feature_names_out(categorical_columns)

# Convert to DataFrame
X_encoded = pd.DataFrame(X_encoded, columns=encoded_feature_names)

# Combine with numerical features
X_numerical = data.drop(columns=[target_column] + categorical_columns)
X = pd.concat([X_numerical, X_encoded], axis=1)
y = data[target_column]

# Check if "model_Focus" exists
if "model_ Focus" in X.columns:
    print("✅ 'model_Focus' found in features.")
else:
    print("❌ 'model_Focus' is missing! Check encoding.")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Decision Tree
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)

# Print Tree Structure
tree_rules = export_text(tree, feature_names=X.columns.to_list())
print("\n==== Decision Tree Structure ====")
# print(tree_rules)

# Feature Importance from Decision Tree
importances = tree.feature_importances_
plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance (Decision Tree)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Check Model Without "Focus"
if "model_ Focus" in X_train.columns:
    X_train_reduced = X_train.drop(columns=["model_ Focus"])
    X_test_reduced = X_test.drop(columns=["model_ Focus"])

    tree_reduced = DecisionTreeRegressor(random_state=42)
    tree_reduced.fit(X_train_reduced, y_train)

    baseline_score = tree.score(X_test, y_test)
    reduced_score = tree_reduced.score(X_test_reduced, y_test)

    print(f"\nDecision Tree R² Score (With 'Focus'): {baseline_score:.4f}")
    print(f"Decision Tree R² Score (Without 'Focus'): {reduced_score:.4f}")

    if abs(baseline_score - reduced_score) < 0.01:
        print("🔍 'Focus' is not contributing much to the model.")
    else:
        print("⚠️ Removing 'Focus' significantly changed the model!")

# Permutation Importance
print("\n==== Running Permutation Importance Test ====")
perm_importance = permutation_importance(tree, X_test, y_test, n_repeats=10, random_state=42)
sorted_idx = np.argsort(perm_importance.importances_mean)

plt.figure(figsize=(12, 6))
plt.barh(np.array(X.columns)[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance Score")
plt.title("Permutation Importance (Decision Tree)")
plt.show()

# Train Random Forest
print("\n==== Training Random Forest for Comparison ====")
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

rf_importances = rf.feature_importances_
plt.figure(figsize=(12, 6))
sns.barplot(x=rf_importances, y=X.columns)
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Compare Feature Importances
df_importance = pd.DataFrame({
    "Feature": X.columns,
    "DecisionTree": importances,
    "RandomForest": rf_importances
}).set_index("Feature")

print("\n==== Feature Importance Comparison ====")
print(df_importance.sort_values("DecisionTree", ascending=False))

# Final Verdict
if "model_ Focus" in df_importance.index:
    if df_importance.loc["model_ Focus", "RandomForest"] < 0.01:
        print("\n🚨 'Focus' is important in Decision Tree but not in Random Forest → Likely Overfitting!")
    else:
        print("\n✅ 'Focus' is important in both models → Likely a real pattern.")
else:
    print("\n❌ 'model_Focus' still missing! Check encoding pipeline.")

# ==========================================================
# Additional Checks
# ==========================================================

# 1️⃣ Correlation Check
print("\n==== Correlation Check ====")
if "model_ Focus" in X.columns:
    correlations = X.corrwith(X["model_ Focus"]).sort_values(ascending=False)
    print(correlations)

    target_corr = X["model_ Focus"].corr(y)
    print(f"\n📊 Correlation between 'model_ Focus' and Target ({target_column}): {target_corr:.4f}")

    # Highlight any strong correlations
    strong_correlations = correlations[abs(correlations) > 0.5]
    if len(strong_correlations) > 0:
        print("\n⚠️ Strong correlations found with:")
        print(strong_correlations)
    else:
        print("\n✅ No strong correlations detected.")

# 2️⃣ Variance Inflation Factor (VIF) Check
print("\n==== Variance Inflation Factor (VIF) Check ====")
if "model_ Focus" in X.columns:
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    high_vif = vif_data[vif_data["VIF"] > 10]
    if not high_vif.empty:
        print("⚠️ High VIF detected, indicating multicollinearity:")
        print(high_vif)
    else:
        print("✅ No multicollinearity detected.")

# 3️⃣ Target Leakage Check
print("\n==== Target Leakage Check ====")
if "model_ Focus" in X.columns:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000)
    
    X_focus = X[["model_ Focus"]]
    clf.fit(X_focus, y > y.median())  # Predict if price is above median
    
    leakage_score = clf.score(X_focus, y > y.median())
    print(f"\n🔍 Predictive Power of 'model_ Focus' on Target: {leakage_score:.4f}")
    
    if leakage_score > 0.9:
        print("🚨 Possible data leakage detected! 'model_ Focus' predicts the target too well.")
    else:
        print("✅ No target leakage detected.")
