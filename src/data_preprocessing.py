import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# define directories

train_dir = 'data/train'
val_dir = 'data/val'

# data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_director(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_director(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)