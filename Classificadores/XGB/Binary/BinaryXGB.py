import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from xgboost import XGBClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE
from joblib import dump


# Load Hu moments from CSV
def load_hu_moments_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    return features, labels


# Load features and labels from CSV
csv_file = "../../../Csvs/hu_moments_modified.csv"
features, labels = load_hu_moments_from_csv(csv_file)

# Plot and save class distribution before oversampling
counter_before = Counter(labels)
plt.figure(figsize=(10, 7))
sns.barplot(x=list(counter_before.keys()), y=list(counter_before.values()))
plt.title("Class Distribution Before Oversampling")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.savefig("Desempenho/class_distribution_before.png")
plt.close()

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42, k_neighbors=2, sampling_strategy="auto")
features_resampled, labels_resampled = smote.fit_resample(features, labels)

# Plot and save class distribution after oversampling
counter_after = Counter(labels_resampled)
plt.figure(figsize=(10, 7))
sns.barplot(x=list(counter_after.keys()), y=list(counter_after.values()))
plt.title("Class Distribution After Oversampling")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.savefig("Desempenho/class_distribution_after.png")
plt.close()

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features_resampled,
    labels_resampled,
    test_size=0.2,
    random_state=42,
    stratify=labels_resampled,
)

# Print the number of positive and negative classes after oversampling
print("Number of positive classes after oversampling:", Counter(y_train)[1])
print("Number of negative classes after oversampling:", Counter(y_train)[0])

# Instantiate the XGBoost classifier
model = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    verbosity=1,
)

# Train the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Save the accuracy to a text file
with open("Desempenho/accuracy.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")

# Plot and save the confusion matrix
plt.figure(figsize=(10, 7))
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
cm_display.plot(cmap=plt.cm.Blues, values_format="d")
plt.title("Confusion Matrix")
plt.savefig("Desempenho/confusion_matrix.png")

# Save the trained model to a pkl file
dump(model, "../../ModelosTreinados/xgboostBinary_model.pkl")
