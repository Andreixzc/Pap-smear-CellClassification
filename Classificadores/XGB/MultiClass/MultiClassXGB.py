import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from collections import Counter

# Load Hu moments from CSV
def load_hu_moments_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    features = df.iloc[:, :-1].values
    labels = df.iloc[:, -1].values
    return features, labels

# Load features and labels from CSV
csvBasePath = 'csv_files/'
csv_file   = csvBasePath + 'hu_moments.csv'
features, labels = load_hu_moments_from_csv(csv_file)

# Encode the class labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Count instances per class before resampling
print("Instances per class before resampling:")
class_counts_before = Counter(encoded_labels)
for class_name, count in zip(label_encoder.classes_, class_counts_before.values()):
    print(f"{class_name}: {count} instances")

# Apply SMOTE to balance the classes
smoteRatio = 1

sampling_strategy = {
    0: 500,  # Example class 0
    1: 500,   # Example class 1
    2: 500,   # Example class 2
    3: 500,   # Example class 3
    4: 902,   # Example class 4
    5: 500    # Example class 5
}
currentDistribution ={
    0: 341,   # ASC-H
    1: 126,   # ASC-US
    2: 321,   # HSIL
    3: 388,   # LSIL
    4: 902,   # Negative for intraepithelial lesion
    5: 43     # SCC
}
def calculateNewDistribution(currentDistribution, smoteRatio, majorityClass):

    newDistribution = {}
    for key in sampling_strategy:
        if key != majorityClass:
            newDistribution[key] = int(sampling_strategy[key] * smoteRatio) + currentDistribution[key]
            if newDistribution[key] > sampling_strategy[majorityClass]:
                newDistribution[key] = sampling_strategy[majorityClass]
    return newDistribution


smote = SMOTE(random_state=42, k_neighbors=2, sampling_strategy=calculateNewDistribution(currentDistribution, smoteRatio, 4))
X_resampled, y_resampled = smote.fit_resample(features, encoded_labels)

# # Count instances per class after resampling
print("\nInstances per class after resampling:")
class_counts_after = Counter(y_resampled)
for class_name, count in zip(label_encoder.classes_, class_counts_after.values()):
    print(f"{class_name}: {count} instances")

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
print(f'\nAccuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

print(confusion_matrix(y_test, y_pred))
