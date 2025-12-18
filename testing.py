import cv2
import joblib
from sklearn.preprocessing import LabelEncoder
from image_loader2 import load_dataset, feature_extraction, extract_combined_features
from image_loader2 import extract_pure_cnn_features
import numpy as np

classifications = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "unknown"]
le = LabelEncoder().fit(classifications)

def predict_with_unknown(model, data, percentage):
    pred = model.predict(data)
    pred_probability = model.predict_proba(data)
    for i in range(len(pred)):
        if max(pred_probability[i]) < percentage:
            pred[i] = le.transform(["unknown"])[0]
    return pred, pred_probability


## Load the best models and use them in the live camera

loaded_svm_model = joblib.load("saved_svm_model.joblib")
loaded_scaler = joblib.load("saved_scaler.joblib")



while True:
    img = cv2.imread(input("Enter Image Path: "))
    features = extract_pure_cnn_features(img)
    features_scaled = loaded_scaler.transform(np.reshape(features, (1, -1)))
    prediction, prediction_probability = predict_with_unknown(loaded_svm_model, features_scaled, 0.6)
    print(prediction_probability, le.inverse_transform(prediction)[0])
