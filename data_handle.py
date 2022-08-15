from config import DataPreparation
from directory_helper import get_filepaths
from shutil import copyfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
import os
import random


def split_data(SOURCE_DIR, TRAINING_DIR, VALIDATION_DIR, SPLIT_SIZE):
    """
    Splits the data into train and test sets

    Args:
      SOURCE_DIR (string): directory path containing the images
      TRAINING_DIR (string): directory path to be used for training
      VALIDATION_DIR (string): directory path to be used for validation
      SPLIT_SIZE (float): proportion of the dataset to be used for training

    Returns:
      None
    """
    list_item = get_filepaths(SOURCE_DIR)
    for item in list(list_item):
        # Only keep true images
        if os.path.getsize(os.path.join(item)) == 0 or not item.endswith(config.IMAGE_EXTENSION):
            print(f'{item} is zero length, so ignoring.')
            list_item.remove(item)

    list_item = [line.split('\\')[-1] for line in list_item]

    list_item_training = random.sample(list_item, int(len(list_item) * SPLIT_SIZE))

    for item in list_item:
        if item not in list_item_training:
            copyfile(os.path.join(SOURCE_DIR, item), os.path.join(VALIDATION_DIR, item))
        else:
            copyfile(os.path.join(SOURCE_DIR, item), os.path.join(TRAINING_DIR, item))


def train_val_generators(TRAINING_DIR, VALIDATION_DIR):
    """
    Creates the training and validation data generators

    Args:
      TRAINING_DIR (string): directory path containing the training images
      VALIDATION_DIR (string): directory path containing the testing/validation images

    Returns:
      train_generator, validation_generator - tuple containing the generators
    """
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=DataPreparation.rotation_range,
                                       width_shift_range=DataPreparation.width_shift_range,
                                       height_shift_range=DataPreparation.height_shift_range,
                                       shear_range=DataPreparation.shear_range,
                                       zoom_range=DataPreparation.zoom_range,
                                       horizontal_flip=DataPreparation.horizontal_flip,
                                       vertical_flip=DataPreparation.vertical_flip,
                                       fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR,
                                                        batch_size=DataPreparation.BATCH_SIZE_TRAIN,
                                                        class_mode='categorical',
                                                        target_size=DataPreparation.SIZE_IMAGE)

    validation_datagen = ImageDataGenerator(rescale=1./255)

    validation_generator = validation_datagen.flow_from_directory(directory=VALIDATION_DIR,
                                                                  batch_size=DataPreparation.BATCH_SIZE_VALID,
                                                                  class_mode='categorical',
                                                                  target_size=DataPreparation.SIZE_IMAGE)
    return train_generator, validation_generator
