import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# 🚀 Load Data
file_path = "../data.csv"  # Update this with your actual file path
df = pd.read_csv(file_path)

# 🔹 Encode Categorical Features (Convert to numbers for correlation analysis)
df_encoded = df.copy()
for col in ["transmission", "model", "fuelType"]:
    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

# 🔹 Compute Correlation Matrix (Only Input Features)
df_input_features = df_encoded.drop(columns=["price"])
correlation_matrix = df_input_features.corr()

# 🔹 Plot Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Feature Correlation Matrix (Independent Variables Only)")
plt.show()

# 🔹 Pairplot of Features
sns.pairplot(df, diag_kind="kde")
plt.show()

# 🔹 Boxplot: Engine Size Distribution by Model
plt.figure(figsize=(12, 6))
sns.boxplot(x=df["model"], y=df["engineSize"])
plt.xticks(rotation=90)  # Rotate model names for better visibility
plt.title("Engine Size Distribution by Model")
plt.show()

# 🔹 Price Distribution by Categorical Features
categorical_columns = ["fuelType", "transmission", "model"]
for cat in categorical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[cat], y=df["price"])
    plt.title(f"Price Distribution by {cat}")
    plt.xticks(rotation=45)
    plt.show()

# 🔹 Boxplots: Model vs. Year, Mileage, and Engine Size
numeric_features = ["year", "mileage", "engineSize"]
plt.figure(figsize=(15, 5))

for i, feature in enumerate(numeric_features, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x=df["model"], y=df[feature])
    plt.title(f"{feature} by Model")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
