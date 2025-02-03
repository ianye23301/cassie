import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.metrics import mean_absolute_error, r2_score
from process import DataProcessor  # Direct import since it's in the same folder

#  Load & Process Data
processor = DataProcessor("../data.csv")
processor.load_data()
processor.split_data()
processor.encode_features()

# Retrieve Preprocessed Data
X_train, X_test, y_train, y_test = processor.get_processed_data()

# Train the Decision Tree
tree = DecisionTreeRegressor(max_depth=10, min_samples_split=2, random_state=42)
tree.fit(X_train, y_train)

# Save the model
joblib.dump(tree, "decision_tree_model.pkl")


# Analyze feature importance
encoder = joblib.load("encoder.pkl")
feature_names = ["year", "mileage", "engineSize"] + list(encoder.get_feature_names_out(["transmission", "model", "fuelType"]))
importances = tree.feature_importances_


plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importance (Decision Tree)")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Save decision rules
tree_rules = export_text(tree, feature_names=feature_names)
print(tree_rules)
with open("decision_tree_rules.txt", "w") as f:
    f.write(tree_rules)

# Test model
y_pred = tree.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

