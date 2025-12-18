import cv2
import joblib
from sklearn.preprocessing import LabelEncoder
from image_loader import extract_pure_cnn_features
import numpy as np
import os

classifications = ["cardboard", "glass", "metal", "paper", "plastic", "trash", "unknown"]
le = LabelEncoder().fit(classifications)

def predict_with_unknown(model, data, percentage):
    pred = model.predict(data)
    pred_probability = model.predict_proba(data)
    for i in range(len(pred)):
        if max(pred_probability[i]) < percentage:
            pred[i] = le.transform(["unknown"])[0]
    return pred, pred_probability

def predict(dataFilePath, bestModelPath):
    # Load the model and scaler
    loaded_model = joblib.load(bestModelPath)
    loaded_scaler = joblib.load("saved_scaler.joblib")
    
    predictions = []
    
    # Get all image files from the folder
    image_files = [f for f in os.listdir(dataFilePath) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    for image_file in image_files:
        image_path = os.path.join(dataFilePath, image_file)
        img = cv2.imread(image_path)
        
        # Extract features
        features = extract_pure_cnn_features(img)
        features_scaled = loaded_scaler.transform(np.reshape(features, (1, -1)))
        
        # Predict
        prediction, prediction_probability = predict_with_unknown(loaded_model, features_scaled, 0.6)
        predicted_class = le.inverse_transform(prediction)[0]
        predictions.append(predicted_class.item())
    
    return predictions