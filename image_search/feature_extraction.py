# vgg19_feature_extraction.py
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Model
from skimage.feature import local_binary_pattern
import numpy as np
import cv2

# Load pre-trained VGG19 model
base_model = VGG19(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Preprocess input image and extract features
def extract_vgg19_features(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features

def extract_lbp_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    radius = 1
    n_points = 8 * radius
    lbp_image = local_binary_pattern(img, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp_image.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)  # Normalize the histogram
    return hist

# Combine VGG and LBP features
def combine_features(vgg_features, lbp_features):
    return np.concatenate((vgg_features, lbp_features), axis=None)
