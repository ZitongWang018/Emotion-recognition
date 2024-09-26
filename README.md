# Emotion Recognition CNN Model

## Overview

This project implements a Convolutional Neural Network (CNN) for emotion recognition, specifically classifying images as either "happy" or "sad". The model is built using TensorFlow and Keras, and is trained on a dataset of grayscale facial images.

## Requirements

- Python 3.x
- NumPy
- Pandas
- TensorFlow
- Matplotlib
- scikit-learn
- Pillow (PIL)

You can install the required packages using pip:

pip install numpy pandas tensorflow matplotlib scikit-learn pillow

## Model Architecture

The CNN model architecture is as follows:

1. Convolutional layer (32 filters, 3x3 kernel)
2. Max pooling layer (2x2)
3. Convolutional layer (64 filters, 3x3 kernel)
4. Max pooling layer (2x2)
5. Flatten layer
6. Dense layer (256 units)
7. Dropout layer (20% dropout rate)
8. Dense layer (64 units)
9. Output dense layer (1 unit, sigmoid activation)

The model is compiled using binary cross-entropy loss and the Adam optimizer.

## Data Preparation

The dataset should be organized in the following structure:

archive/
├── train/
│   ├── happy/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── sad/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
└── test/
├── happy/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── sad/
├── image1.jpg
├── image2.jpg
└── ...

Images are loaded, resized to 48x48 pixels, converted to grayscale, and normalized.

## Training

The model is trained on the prepared dataset with the following parameters:

- Epochs: 10
- Batch size: 64
- Validation split: 20%

## Evaluation

The model's performance is evaluated on a separate test set. The evaluation metrics include loss and accuracy.

## Prediction

The trained model can be used to predict the emotion (happy or not happy) of new images. The prediction process includes:

1. Loading and preprocessing the image
2. Making a prediction using the trained model
3. Interpreting the result (threshold of 0.5 for binary classification)

## Visualization

The project includes functionality to visualize:

1. Training history (accuracy and loss over epochs)
2. Sample predictions on test images
3. Predictions on new, unseen images

## Usage

1. Prepare your dataset in the required folder structure.
2. Run the training script to train the model.
3. Evaluate the model on the test set.
4. Use the trained model to make predictions on new images.

Example usage for prediction:

```python
image_path = 'path/to/your/image.jpg'
image = load_and_preprocess_image(image_path)
prediction = model.predict(image)
predicted_label = int(prediction > 0.5)
print(f"Prediction: {'Happy' if predicted_label == 1 else 'Not Happy'} ({prediction[0][0]:.2f})")
```

Future Improvements

* Experiment with different model architectures
* Use data augmentation to improve model generalization
* Implement multi-class classification for more emotion categories
* Fine-tune hyperparameters for better performance
