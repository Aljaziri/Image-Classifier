import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import argparse
import json

# Constants
BATCH_SIZE = 32
IMAGE_SIZE = 224

# Image processing function
def preprocess_image(image): 
    processed_image = tf.cast(image, tf.float32)
    processed_image = tf.image.resize(processed_image, (IMAGE_SIZE, IMAGE_SIZE))
    processed_image /= 255
    return processed_image.numpy()

# Prediction function
def make_predictions(image_path, model, top_k=5):
    input_image = Image.open(image_path)
    input_array = np.asarray(input_image)
    processed_array = preprocess_image(input_array)
    final_input = np.expand_dims(processed_array, axis=0)
    
    predictions = model.predict(final_input)
    top_k_probabilities = -np.partition(-predictions[0], top_k)[:top_k]
    top_k_indices = np.argpartition(-predictions[0], top_k)[:top_k]
    
    return top_k_probabilities, top_k_indices

if __name__ == '__main__':
    # Command-line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('image_path')
    parser.add_argument('model_path')
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--category_names') 
    
    args = parser.parse_args()
    print(args)
    
    print('prediction_script.py is executing')
    print('image_path:', args.image_path)
    print('model_path:', args.model_path)
    print('top_k:', args.top_k)
    print('category_names:', args.category_names)
    
    # Assign command-line arguments to variables
    input_path = args.image_path
    
    # Load the pre-trained model
    trained_model = tf.keras.models.load_model(args.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    top_k_predictions = args.top_k
        
    # Make predictions using the make_predictions function
    top_k_probs, top_k_indices = make_predictions(input_path, trained_model, top_k_predictions)

    if args.category_names:
        # Load class names mapping from JSON file
        with open(args.category_names, 'r') as file:
            class_names_mapping = json.load(file)
        # Adjust class index to start from 1 and convert to string
        class_names_mapping = {str(int(key)+1): value for key, value in class_names_mapping.items()}
        # Map class indices to class names
        top_k_class_names = [class_names_mapping[str(idx+1)] for idx in top_k_indices]
    
    # Print results
    print('_' * 70)
    print('Probabilities:', top_k_probs)
    print('Top {} Indices:'.format(top_k_predictions), top_k_indices)
    if args.category_names:
        print('Top {} Class Names:'.format(top_k_predictions), top_k_class_names)
    print('_' * 70)
