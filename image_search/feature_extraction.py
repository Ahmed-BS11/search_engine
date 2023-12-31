# vgg19_feature_extraction.py
from tensorflow import keras
from keras.models import Model
from keras.applications.vgg19 import VGG19, preprocess_input

from skimage.feature import local_binary_pattern
import numpy as np
import cv2
import os
import pandas as pd


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


def get_image_files(root_dir):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']  
    image_files = []

    for category in os.listdir(root_dir):
        category_path = os.path.join(root_dir, category)
        print(category_path)
        if os.path.isdir(category_path):
            for root, _, files in os.walk(category_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        image_files.append(os.path.join(root, file))

    return image_files

root_directory = r"C:\Users\ahmed\Desktop\Supcom\INDP3_AIM\cbir\bdimage\image_db" 
image_files = get_image_files(root_directory)

final_vector_list=[]

for image_path in image_files[:5]:
    vgg_features=extract_vgg19_features(image_path)
    lbp_features=extract_lbp_features(image_path)
    final_vector=combine_features(vgg_features, lbp_features)
    final_vector_list.append(final_vector)



feature_vector_df = pd.DataFrame(final_vector_list)
feature_vector_df.to_csv("feature_vectors.csv", index=False)

