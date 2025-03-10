from tensorflow.keras.models import load_model
from .data_preprocessing import val_generator

# Load the saved model
model = load_model('models/fruit_cnn_model.h5')

# Evaluate the model
loss, accuracy = model.evaluate(val_generator)
print(f"Validation Loss: {loss:.4f}")
print(f"Validation Accuracy: {accuracy:.4f}")