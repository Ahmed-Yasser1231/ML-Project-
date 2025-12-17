from image_loader import load_dataset, feature_extraction, extract_combined_features
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
# load images into a data, ensure augementation is only in training data

def predict_with_unknown(model, data, percentage):
    pred = model.predict(data)
    pred_probability = model.predict_proba(data)
    for i in range(len(pred)):
        if max(pred_probability[i]) < percentage:
            pred[i] = le.transform(["unknown"])[0]
    return pred, pred_probability


dataset_path = "dataset/"
classifications = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "unknown"]


le = LabelEncoder().fit(classifications)
scaler = StandardScaler()

# Load dataset with augmentation to balance classes to ~500 samples each
dataset = load_dataset(dataset_path, classifications, target_per_class=500)

X = np.array(dataset['features'].tolist())
print(f"Features: {len(X[0])}")
y = le.transform(dataset["classification"])

# Split dataset - augmentation already applied to training data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features - IMPORTANT for SVM
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

## Create the models with hyperparameter tuning
print("\nPerforming hyperparameter tuning...")
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf', 'poly']
}

svm_model = GridSearchCV(SVC(probability=True, random_state=42, class_weight='balanced'), 
                            param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

# knn_model = KNeighborsClassifier()
## Use cross validation on the current model and data (multiple ways)

# svm_cv_scores = cross_val_score(svm_model, X, y, cv=5)
# knn_cv_scores = cross_val_score(knn_model, X, y, cv=5)
# print(svm_cv_scores)
# print(knn_cv_scores)

## Fit the model

svm_model.fit(X_train_scaled, y_train)
print(f"\nBest parameters: {svm_model.best_params_}")
print(f"Best cross-validation score: {svm_model.best_score_:.4f}")

# knn_model.fit(X_train, y_train)

## Test the models and check it's accuracy

svm_pred, svm_pred_probability = predict_with_unknown(svm_model, X_test_scaled, 0.6)
# knn_pred = knn_model.predict(X_test)
print(svm_pred_probability)
results = pd.DataFrame()
results["real"] = le.inverse_transform(y_test)
results["svm_predictions"] = le.inverse_transform(svm_pred)
# results["knn_pred"] = le.inverse_transform(knn_pred)
print(results)

svm_accuracy = accuracy_score(y_test, svm_pred)
# knn_accuracy = accuracy_score(y_test, knn_pred)

# Print accuracy of both models
print(f"SVM Accuracy: {svm_accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, svm_pred, target_names=le.inverse_transform(range(len(classifications)))))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svm_pred))

# print(f"KNN Accuracy: {knn_accuracy * 100:.2f}%")
## Save the best models as a file
joblib.dump(svm_model, "saved_svm_model.joblib")
joblib.dump(scaler, "saved_scaler.joblib")

