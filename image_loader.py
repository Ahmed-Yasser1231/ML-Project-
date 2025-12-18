import cv2
import pandas as pd
import os
import numpy as np
from skimage.feature import hog
from skimage import color
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Global CNN model for feature extraction (loaded once to avoid reloading)
_cnn_model = None

def get_cnn_feature_extractor():
    """Load and cache pre-trained MobileNetV2 model for feature extraction"""
    global _cnn_model
    if _cnn_model is None:
        print("Loading pre-trained MobileNetV2 model...")
        # Load MobileNetV2 pre-trained on ImageNet
        # include_top=False removes the final classification layer
        # pooling='avg' adds global average pooling to get fixed-size features
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False  # Freeze weights for feature extraction
        _cnn_model = base_model
        print("MobileNetV2 model loaded successfully!")
    return _cnn_model

## Data Augmentation Functions
def rotate_image(image, angle):
    """Rotate image by specified angle"""
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated

def flip_image(image, flip_code):
    """Flip image horizontally (1), vertically (0), or both (-1)"""
    return cv2.flip(image, flip_code)

def scale_image(image, scale_factor):
    """Scale image by specified factor"""
    height, width = image.shape[:2]
    new_height, new_width = int(height * scale_factor), int(width * scale_factor)
    scaled = cv2.resize(image, (new_width, new_height))
    # Crop or pad to original size
    if scale_factor > 1:
        # Crop center
        start_y = (new_height - height) // 2
        start_x = (new_width - width) // 2
        scaled = scaled[start_y:start_y+height, start_x:start_x+width]
    else:
        # Pad with zeros
        pad_y = (height - new_height) // 2
        pad_x = (width - new_width) // 2
        scaled = cv2.copyMakeBorder(scaled, pad_y, height-new_height-pad_y, 
                                    pad_x, width-new_width-pad_x, 
                                    cv2.BORDER_CONSTANT, value=[0,0,0])
    return scaled

def adjust_brightness(image, factor):
    """Adjust image brightness"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:,:,2] = hsv[:,:,2] * factor
    hsv[:,:,2][hsv[:,:,2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def add_noise(image, noise_level=25):
    """Add Gaussian noise to image"""
    noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def adjust_contrast(image, factor):
    """Adjust image contrast"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = np.clip(l * factor, 0, 255).astype(np.uint8)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def random_crop_and_resize(image, crop_factor=0.8):
    """Randomly crop and resize back to original size"""
    height, width = image.shape[:2]
    new_height, new_width = int(height * crop_factor), int(width * crop_factor)
    
    max_x = width - new_width
    max_y = height - new_height
    
    x = np.random.randint(0, max_x + 1)
    y = np.random.randint(0, max_y + 1)
    
    cropped = image[y:y+new_height, x:x+new_width]
    resized = cv2.resize(cropped, (width, height))
    return resized

def augment_image(image, augmentation_type):
    """Apply specific augmentation technique"""
    if augmentation_type == 'rotate_15':
        return rotate_image(image, 15)
    elif augmentation_type == 'rotate_-15':
        return rotate_image(image, -15)
    elif augmentation_type == 'rotate_30':
        return rotate_image(image, 30)
    elif augmentation_type == 'rotate_-30':
        return rotate_image(image, -30)
    elif augmentation_type == 'flip_horizontal':
        return flip_image(image, 1)
    elif augmentation_type == 'flip_vertical':
        return flip_image(image, 0)
    elif augmentation_type == 'scale_0.8':
        return scale_image(image, 0.8)
    elif augmentation_type == 'scale_1.2':
        return scale_image(image, 1.2)
    elif augmentation_type == 'brightness_0.7':
        return adjust_brightness(image, 0.7)
    elif augmentation_type == 'brightness_1.3':
        return adjust_brightness(image, 1.3)
    elif augmentation_type == 'contrast_0.8':
        return adjust_contrast(image, 0.8)
    elif augmentation_type == 'contrast_1.2':
        return adjust_contrast(image, 1.2)
    elif augmentation_type == 'noise':
        return add_noise(image, 25)
    elif augmentation_type == 'crop':
        return random_crop_and_resize(image, 0.85)
    else:
        return image

## Load Images
## Data Augmentation (multiple techniques) ensure augementation is only in training data
## Extract Features and set Classification from Image Path (multiple feature extraction techniques)
## Store the results in a DataFrame as the data used for training and testing
def load_dataset(path, classifications, target_per_class=500):
    """
    Load dataset and balance classes to target_per_class using data augmentation
    """
    data = []
    n = 0
    
    # Augmentation techniques to use
    augmentation_techniques = [
        'rotate_15', 'rotate_-15', 'rotate_30', 'rotate_-30',
        'flip_horizontal', 'flip_vertical',
        'scale_0.8', 'scale_1.2',
        'brightness_0.7', 'brightness_1.3',
        'contrast_0.8', 'contrast_1.2',
        'noise', 'crop'
    ]
    
    for classification in classifications:
        if classification == "unknown":
            continue
        
        original_images = []
        
        # Load all original images for this class
        with os.scandir(path + classification) as entries:
            for entry in entries:
                n += 1
                print(f"Loading original image {n}: {classification}")
                img = cv2.imread(entry.path)
                if img is not None:
                    original_images.append(img)
        
        print(f"Loaded {len(original_images)} original images for {classification}")
        
        # Add original images to dataset
        for img in original_images:
            features = extract_pure_cnn_features(img)
            new_row = [False, features, classification]
            data.append(new_row)
        
        # Calculate how many augmented images we need
        num_originals = len(original_images)
        num_augmented_needed = target_per_class - num_originals
        
        if num_augmented_needed > 0:
            print(f"Generating {num_augmented_needed} augmented images for {classification}")
            augmented_count = 0
            
            # Generate augmented images
            while augmented_count < num_augmented_needed:
                for img in original_images:
                    if augmented_count >= num_augmented_needed:
                        break
                    
                    # Select random augmentation technique
                    aug_type = np.random.choice(augmentation_techniques)
                    augmented_img = augment_image(img, aug_type)
                    
                    features = extract_pure_cnn_features(augmented_img)
                    new_row = [True, features, classification]
                    data.append(new_row)
                    augmented_count += 1
                    
                    print(f"Augmented {augmented_count}/{num_augmented_needed} for {classification}")

    dataset = pd.DataFrame(data, columns=['is_augmented_data', 'features', 'classification'])
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(dataset)}")
    print(f"Original samples: {len(dataset[dataset['is_augmented_data'] == False])}")
    print(f"Augmented samples: {len(dataset[dataset['is_augmented_data'] == True])}")
    print("\nClass distribution:")
    print(dataset['classification'].value_counts())
    
    return dataset
    
def feature_extraction(image):
    #img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(image, (24, 24))
    features = np.array(img_resized.flatten())
    ## Preprocess the features (multiple preprocessing techniques)
    features = features / 255
    return features

def extract_hog_features(image, resize_dim=(256, 256), pixels_per_cell=(16, 16), cells_per_block=(2, 2), orientations=12):
    """Extract HOG features with optimized parameters"""
    # Step 1: Resize the image
    image_resized = cv2.resize(image, resize_dim)
    
    # Step 2: Denoise the image
    image_denoised = cv2.fastNlMeansDenoisingColored(image_resized, None, 10, 10, 7, 21)
    
    # Step 3: Convert to grayscale
    image_gray = cv2.cvtColor(image_denoised, cv2.COLOR_BGR2GRAY)
    
    # Step 4: Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_gray = clahe.apply(image_gray)
    
    # Step 5: Compute HOG features
    features = hog(image_gray, orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   visualize=False,
                   feature_vector=True,
                   block_norm='L2-Hys')
    
    return features

def extract_cnn_features(image):
    """
    Extract deep learning features using pre-trained CNN (MobileNetV2).
    The CNN automatically learns hierarchical features:
    - Low level: edges, textures, colors
    - Mid level: shapes, patterns
    - High level: object parts, complex structures
    """
    model = get_cnn_feature_extractor()
    
    # Resize to MobileNetV2 input size (224x224)
    img_resized = cv2.resize(image, (224, 224))
    
    # Convert BGR (OpenCV) to RGB (Keras)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
    img_array = np.expand_dims(img_rgb, axis=0)
    
    # Preprocess for MobileNetV2 (normalize to [-1, 1])
    img_preprocessed = preprocess_input(img_array)
    
    # Extract features (output shape: (1, 1280))
    features = model.predict(img_preprocessed, verbose=0)
    
    # Flatten to 1D array
    return features.flatten()

def extract_pure_cnn_features(image):
    """
    Extract ONLY CNN features (1280 features).
    Use this for a pure deep learning approach.
    Faster and simpler than combined features.
    """
    return extract_cnn_features(image)