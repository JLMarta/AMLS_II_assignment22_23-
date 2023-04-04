import os
import glob
import cv2
import numpy as np
import seaborn as sns
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import tensorflow as tf

def input_generatation(path):
    image_gen = ImageDataGenerator()

    train_generator = image_gen.flow_from_directory(
    path + 'Train/',
    target_size=(200,200),
    batch_size=32,
    color_mode='rgb',
    class_mode='binary')

    valid_generator = image_gen.flow_from_directory(
    path + 'Validation/',
    target_size=(200, 200),
    batch_size=32,
    color_mode='rgb',
    class_mode='binary')

    test_generator = image_gen.flow_from_directory(
    path + 'Test/',
    target_size=(200, 200),
    batch_size=1,
    color_mode='rgb',
    shuffle = False,
    class_mode='binary')

    return train_generator, valid_generator, test_generator



def densenet_model():
    densenet = DenseNet121( weights=None, include_top=False, input_shape=(200,200,3) )
    model = Sequential([ 
                densenet,
                layers.GlobalAveragePooling2D(),
                layers.Dense(1, activation='sigmoid')
            ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def densenet_train(model, train_generator, valid_generator):
    history = model.fit(
    train_generator,
    steps_per_epoch = (27985//32),
    validation_data = valid_generator,
    validation_steps = (3498//32),
    epochs = 40)

    return history

def plot_train_valid(history, name):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.ylim([0,1.0])
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    # plt.show()
    plt.savefig(name)
    print('Train & Valid Curve is Saved.')

def densenet_test(model, test_generator, figpath):
    y_pred = model.predict(test_generator)
    y_test = test_generator.classes
    plt.figure(figsize = (8,5))
    sns.heatmap(metrics.confusion_matrix(y_test, y_pred.round()), annot = True,fmt="d",cmap = "Blues")
    # plt.show()
    plt.savefig(figpath)
    print('Test figure is saved')
    print("Accuracy Score:", metrics.accuracy_score(y_test, y_pred.round()))
    print("ROC-AUC Score:", metrics.roc_auc_score(y_test, y_pred))
    print("AP Score:", metrics.average_precision_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred > 0.5))


def cnn_test(model, path, figpath):
    image_gen = ImageDataGenerator()

    test_generator = image_gen.flow_from_directory(
    path + 'Test/',
    target_size=(200, 200),
    batch_size=1,
    color_mode='rgb',
    shuffle = False,
    class_mode='binary')

    y_pred = model.predict(test_generator)
    y_test = test_generator.classes
    plt.figure(figsize = (8,5))
    sns.heatmap(metrics.confusion_matrix(y_test, y_pred.round()), annot = True,fmt="d",cmap = "Blues")
    # plt.show()
    plt.savefig(figpath)
    print('Test figure is saved')
    print("Accuracy Score:", metrics.accuracy_score(y_test, y_pred.round()))
    print("ROC-AUC Score:", metrics.roc_auc_score(y_test, y_pred))
    print("AP Score:", metrics.average_precision_score(y_test, y_pred))
    print(metrics.classification_report(y_test, y_pred > 0.5))
