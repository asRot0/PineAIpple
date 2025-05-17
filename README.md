# PineAIpple

This project is focused on building a deep learning model to classify fruits based on images using Convolutional Neural Networks (CNNs).

## Dataset

The dataset used in this project is the **Fruits-360** dataset. It contains images of various fruits that are grouped into different categories. The dataset is divided into three folders:
- **Training data**: Contains images for training the model.
- **Validation data**: Used to validate the model during training.
- **Test data**: Used to evaluate the model after training.

You can download the full dataset from [Fruits-360 Dataset on Kaggle](https://www.kaggle.com/datasets/moltean/fruits).

## Project Overview

- Preprocess and augment the images using TensorFlow’s `ImageDataGenerator`.
- Define a Convolutional Neural Network (CNN) model for image classification.
- Train the model using the training and validation sets.
- Evaluate the model using the test set to measure its accuracy.
- Visualize results including prediction examples.

### Steps
1. **Data Preprocessing**:
   - Augment the images (rotate, shift, zoom, etc.) to increase dataset variety.
   - Split the data into training, validation, and test sets.
   
2. **Model Definition**:
   - A CNN model is built using TensorFlow/Keras layers to recognize fruit images.

3. **Training**:
   - The model is trained using the augmented data and validation sets.
   - Early stopping and model checkpointing are used to avoid overfitting.

4. **Evaluation**:
   - The model is evaluated using the test dataset, and performance metrics are displayed.

5. **Prediction Visualization**:
   - Predictions are made on test images, and the results are visualized.

## Technologies Used

- **Python**: Programming language used for the project.
- **TensorFlow/Keras**: For building and training the CNN model.
- **Matplotlib/Seaborn**: For data visualization (plots and graphs).
- **Scikit-learn**: For additional evaluation metrics like classification report.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- Seaborn
- Scikit-learn

### Install Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```