import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter
from joblib import dump

# Load Hu moments from CSV
def load_hu_moments_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    return features, labels

# Load features and labels from CSV
csv_file = '../../../Csvs/hu_moments.csv'
features, labels = load_hu_moments_from_csv(csv_file)

# Encode the class labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Count instances per class before resampling
print("Instances per class before resampling:")
class_counts_before = Counter(encoded_labels)
for class_name, count in zip(label_encoder.classes_, class_counts_before.values()):
    print(f"{class_name}: {count} instances")

# Create directories if they do not exist
output_dir = 'Desempenho'
os.makedirs(output_dir, exist_ok=True)

# Plot and save class distribution before oversampling
plt.figure(figsize=(10, 7))
sns.barplot(x=list(class_counts_before.keys()), y=list(class_counts_before.values()))
plt.title('Class Distribution Before Oversampling')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_dir, 'class_distribution_before.png'))
plt.close()

# Apply SMOTE to balance the classes with 'auto' strategy
smote = SMOTE(random_state=42, k_neighbors=2, sampling_strategy='auto')
X_resampled, y_resampled = smote.fit_resample(features, encoded_labels)

# Count instances per class after resampling
print("\nInstances per class after resampling:")
class_counts_after = Counter(y_resampled)
for class_name, count in zip(label_encoder.classes_, class_counts_after.values()):
    print(f"{class_name}: {count} instances")

# Plot and save class distribution after oversampling
plt.figure(figsize=(10, 7))
sns.barplot(x=list(class_counts_after.keys()), y=list(class_counts_after.values()))
plt.title('Class Distribution After Oversampling')
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.savefig(os.path.join(output_dir, 'class_distribution_after.png'))
plt.close()

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Train an XGBoost classifier
classifier = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=4,
    verbosity=1
)
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Print the classification results
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAccuracy: {accuracy}')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# Save the accuracy to a text file
with open(os.path.join(output_dir, 'accuracy.txt'), 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")

# Update label names
label_names = list(label_encoder.classes_)
label_names[label_names.index('Negative for intraepithelial lesion')] = 'negative'

# Plot and save the confusion matrix
plt.figure(figsize=(10, 7))
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=label_names)
cm_display.plot(cmap=plt.cm.Blues, values_format='d')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))

# Save the trained model to a pkl file
model_dir = '../../ModelosTreinados'
os.makedirs(model_dir, exist_ok=True)
dump(classifier, os.path.join(model_dir, 'xgboostMulti_model.pkl'))
