from .model import build_cnn_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from .data_preprocessing import train_generator, val_generator

# Build the model
model = build_cnn_model(num_classes=len(train_generator.class_indices))

# Callbacks
checkpoint = ModelCheckpoint('models/fruit_cnn_model.h5', save_best_only=True, monitor='val_accuracy')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    epochs=20,
    verbose=2,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)