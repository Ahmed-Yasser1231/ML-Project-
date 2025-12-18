import cv2
import joblib
from sklearn.preprocessing import LabelEncoder
from image_loader import extract_pure_cnn_features
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



loaded_svm_model = joblib.load("saved_svm_model.joblib")
loaded_knn_model = joblib.load("saved_knn_model.joblib")
loaded_scaler = joblib.load("saved_scaler.joblib")

# Start capturing video from the webcam (0 is the default camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame from camera

    if not ret:
        print("Failed to grab frame.")
        break

    features = extract_pure_cnn_features(frame)
    features_scaled = loaded_scaler.transform(np.reshape(features, (1, -1)))

    # Make prediction using the model
    svm_prediction, svm_prediction_probability = predict_with_unknown(loaded_svm_model, features_scaled, 0.6)
    knn_prediction, knn_prediction_probability = predict_with_unknown(loaded_knn_model, features_scaled, 0.6)
    # prediction = loaded_svm_model.predict(np.reshape(features, (1, -1)))
    # prediction_probability = max(loaded_svm_model.predict_proba(np.reshape(features, (1, -1))))
    # if max(prediction_probability) < 0.6:
    #     prediction[0] = len(classifications) - 1
    print(svm_prediction_probability, le.inverse_transform(svm_prediction)[0] , max(svm_prediction_probability[0]))
    print(knn_prediction_probability, le.inverse_transform(knn_prediction)[0] , max(knn_prediction_probability[0]))

    cv2.putText(frame, f'Prediction: {le.inverse_transform(svm_prediction)[0]}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f'Prediction: {le.inverse_transform(knn_prediction)[0]}', (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the live camera feed with the prediction
    cv2.imshow('Live Camera Feed', frame)

    # If the user presses 'q', exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()