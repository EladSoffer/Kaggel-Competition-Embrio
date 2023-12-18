import cv2
import os
import numpy as np
from keras.src.applications import VGG16, DenseNet121
from keras.src.callbacks import Callback
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.applications import MobileNet
import tensorflow as tf
import tensorflow
keras = tf.keras
K = keras.backend
KL = keras.layers
Lambda, Input, Flatten = KL.Lambda, KL.Input, KL.Flatten
Model = keras.Model
from keras import layers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
# Step 1: Data Preparation
import pandas as pd
from keras.applications import ResNet50
from keras.layers import GlobalAveragePooling2D, Dense, Flatten, Conv2D, MaxPooling2D, Input
# Load the ResNet50 model without the top (final) classification layers
# Create the ResNet50 base model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Create a Sequential model for your custom classification layers
custom_layers = Sequential([
    #Conv2D(32, (3, 3), activation='relu', input_shape=(5, 5, 2048)),  # Your Conv2D layer
    GlobalAveragePooling2D(),  # Global Average Pooling Layer
    #MaxPooling2D((2, 2)),  # MaxPooling2D layer
    #Flatten(),
    Dense(1, activation='sigmoid')  # Output layer for your specific task
])

# Connect the output of the base model to the custom classification layers
model6 = Sequential()
model6.add(base_model)
model6.add(custom_layers)
# Load VGG16 with pre-trained weights and without the top (fully connected) layers
#base_model = VGG16(weights='imagenet', include_top=False)
# Add your custom classification layers on top of the base model
#x = GlobalAveragePooling2D()(base_model.output)
#predictions = Dense(1, activation='sigmoid')(x)
# Add more custom layers for classification as needed

# Create your custom model
#custom_model = Model(inputs=base_model.input, outputs=x)
#modelll = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=True)

from keras.layers import GlobalAveragePooling2D, Dense, Dropout

#x = (modelll.output)
#x = Dense(256, activation='relu')(x)
#x = Dropout(0.5)(x)
#predictions = Dense(1, activation='sigmoid')(x)
from keras.models import Model

#custom_model = Model(inputs=modelll.input, outputs=predictions)
#custom_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Add custom layers for your specific task
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#x = Dense(256, activation='relu')(x)  # Add your custom layers
#predictions = Dense(1, activation='sigmoid')(x)  # Output layer for binary classification

#modell = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
#modell.compile(optimizer=SGD(lr=0.01, momentum=0.9),  # Choose your optimizer and learning rate
 #             loss='binary_crossentropy',  # Choose your loss function
  #            metrics=['accuracy'])

# Read labels from csv

#labels_df = pd.read_csv('hvwc23/train.csv')

dataset_path = 'hvwc23/train/'  # Change this to your dataset path
good_images_dir = 'good_images/'
bad_images_dir = 'bad_images/'
good_images_after_aug_dir = 'good_images_after_aug/'
bad_images_after_aug_dir = 'bad_images_after_aug/'

# Create the directories if they don't exist
os.makedirs(good_images_dir, exist_ok=True)
os.makedirs(good_images_after_aug_dir, exist_ok=True)
os.makedirs(bad_images_dir, exist_ok=True)
os.makedirs(bad_images_after_aug_dir, exist_ok=True)

# Read labels from the CSV file
#labels_df = pd.read_csv('hvwc23/train.csv')

#Loop through the dataset and move images to the appropriate directories
#for index, row in labels_df.iterrows():
#    image_file = row['Image']
#    label = row['Class']
#    image_path = os.path.join(dataset_path, image_file)

#    if label == 1:
        # Move the "good" image to the "good_images" directory
#        shutil.copy(image_path, os.path.join(good_images_dir, image_file))
#    else:
        # Move the "bad" image to the "bad_images" directory
 #       shutil.copy(image_path, os.path.join(bad_images_dir, image_file))

# Extract image file names and labels
#image_file_names = labels_df['Image'].tolist()
#labels = labels_df['Class'].tolist()

dataset_path = 'hvwc23/train/'

# # Lists to store images and labels
images = []
labels = []
#
# for image_file, label in zip(image_file_names, labels):
#     image_path = os.path.join(dataset_path, image_file)
#     image = cv2.imread(image_path)
#     image = cv2.resize(image, (150, 150))  # Resize images to a common size
#     image = image / 255.0  # Normalize pixel values to [0, 1]
#     images.append(image)
#     label_list.append(label)
#
 #images = np.array(images)
#labels = np.array(label_list)

good_images = []
bad_images = []
good_images_after_aug = []
bad_images_after_aug = []

# # For "good" images
#for filename in os.listdir(good_images_dir):
#    if filename.endswith(".jpg"):
#        image_path = os.path.join(good_images_dir, filename)
#        image = cv2.imread(image_path)
#        image = cv2.resize(image, (150, 150))  # Resize images to a common size
#        image = image / 255.0  # Normalize pixel values to [0, 1]
#        good_images.append(image)
#
# # For "bad" images
#for filename in os.listdir(bad_images_dir):
#    if filename.endswith(".jpg"):
#        image_path = os.path.join(bad_images_dir, filename)
#        image = cv2.imread(image_path)
#        image = cv2.resize(image, (150, 150))  # Resize images to a common size
#        image = image / 255.0  # Normalize pixel values to [0, 1]
#        bad_images.append(image)

# Convert lists to NumPy arrays if needed
good_images = np.array(good_images)
bad_images = np.array(bad_images)
good_images_after_aug = np.array(good_images_after_aug)
bad_images_after_aug = np.array(bad_images_after_aug)

# Calculate the number of times to augment the "good" images
# target_multiple = 5.77
# num_good_images = len(good_images)
# num_augmentations = int(num_good_images * target_multiple) - num_good_images
#
num_augmentations_good = 1000
num_augmentations_bad = 408
# Initialize an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


# #Create augmented images and save them to the output directory
#for i in range(num_augmentations_good):
#    for x in datagen.flow(good_images, batch_size=1):
#        # Save the augmented image to the output directory
#        output_filename = f"augmented_{i}.jpg"
#        output_path = os.path.join(good_images_after_aug_dir, output_filename)
#        cv2.imwrite(output_path, (x[0] * 255).astype(np.uint8))
#        break  # Exit the loop after saving one augmented image

#Create augmented images and save them to the output directory
#for i in range(num_augmentations_bad):
#    for x in datagen.flow(bad_images, batch_size=1):
#        # Save the augmented image to the output directory
#        output_filename = f"augmented_{i}.jpg"
#        output_path = os.path.join(bad_images_after_aug_dir, output_filename)
#        cv2.imwrite(output_path, (x[0] * 255).astype(np.uint8))
#        break  # Exit the loop after saving one augmented image

# Load and preprocess "good" images
count = 0
for filename in os.listdir(good_images_after_aug_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(good_images_after_aug_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (150, 150))  # Resize images to a common size
        image = image / 255.0  # Normalize pixel values to [0, 1]
        images.append(image)
        labels.append(1)  # Label for "good" images is 1

# Load and preprocess "bad" images
for filename in os.listdir(bad_images_after_aug_dir):
    if filename.endswith(".jpg"):
        image_path = os.path.join(bad_images_after_aug_dir, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (150, 150))  # Resize images to a common size
        image = image / 255.0  # Normalize pixel values to [0, 1]
        images.append(image)
        labels.append(0)  # Label for "bad" images is 0

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=0)


# Step 2: Define the Neural Network Model
model = Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
])

model2 = Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    # # Fully connected layers
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
])

model3 = Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.BatchNormalization(),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
])

model4 = Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
])
model5 = Sequential([
    # Convolutional layers
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')  # Output layer with 1 neuron for binary classification
])



# Step 3: Compile the Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model4.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model5.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
model6.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])


train_data_generator = ImageDataGenerator(
    #rescale=1./255,      # Normalize pixel values
    #rotation_range=20,   # Randomly rotate images by up to 20 degrees
    #width_shift_range=0.2,  # Randomly shift the width of images
    #height_shift_range=0.2,  # Randomly shift the height of images
    #horizontal_flip=True  # Randomly flip images horizontally
)

# Create data generators for training and validation
#train_generator = train_data_generator.flow(train_data, train_labels, batch_size=32)
#validation_generator = train_data_generator.flow(validation_data, validation_labels, batch_size=16)

# Step 5: Training
# Train the model using the data generators
class_weights1 = {0: 2.15, 1: 1}  # Adjust the weight for class 1 based on the imbalance
class_weights2 = {0: 1, 1: 2.2}  # Adjust the weight for class 1 based on the imbalance
class_weights3 = {0: 1, 1: 1}
class_weights4 = {0: 2.5, 1: 1}
class_weights5 = {0: 3.2, 1: 1}
# Define the EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor the validation loss
    patience=5,  # Number of epochs with no improvement after which training will be stopped
    min_delta=0.001,
    restore_best_weights=True  # Restore the model weights from the epoch with the best validation loss
)
class ConfusionMatrixCallback(Callback):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.X)
        true_labels = self.y
        predicted_labels = (predictions >= 0.5).astype(int)
        confusion = confusion_matrix(true_labels, predicted_labels)

        print("Confusion Matrix for Epoch", epoch + 1)
        print(confusion)

# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     batch_size=32,
#     epochs=10,
#     class_weight=class_weights1,
#     callbacks=[early_stopping, ConfusionMatrixCallback(X_train, y_train)],
# )
#
# history = model2.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     batch_size=32,
#     epochs=10,
#     class_weight=class_weights2,
#     callbacks=[early_stopping, ConfusionMatrixCallback(X_train, y_train)],
# )
#
#history = model3.fit(
#    X_train, y_train,
#    validation_data=(X_val, y_val),
#    batch_size=8,
#    epochs=10,
#    class_weight=class_weights3,
#    callbacks=[early_stopping, ConfusionMatrixCallback(X_train, y_train)],
#)

#history = model4.fit(
#    X_train, y_train,
#    validation_data=(X_val, y_val),
#    batch_size=64,
#    epochs=10,
#    class_weight=class_weights4,
#    callbacks=[early_stopping, ConfusionMatrixCallback(X_train, y_train)],
#)

# history = model5.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     batch_size=32,
#     epochs=10,
#     class_weight=class_weights5,
#     callbacks=[early_stopping, ConfusionMatrixCallback(X_train, y_train)],
# )

history = model6.fit(X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=10,
    #class_weight=class_weights5,
    callbacks=[early_stopping, ConfusionMatrixCallback(X_train, y_train)],
)

# Calculate and print the confusion matrix for the training data
# y_pred = model.predict(X_train)
# y_pred_binary = (y_pred >= 0.5).astype(int)
# confusion_matrix_train = confusion_matrix(y_train, y_pred_binary)
# print("Confusion Matrix for Training Data:")
# print(confusion_matrix_train)

# Train the model using the data generators and class weights, while adding early stopping
#model.fit(train_generator, validation_data=validation_generator, epochs=10, callbacks=[early_stopping])
# Train the model
#model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Convert the model's predictions on the validation set to binary labels (0 or 1)
# predicted_labels = [1 if x >= 0.5 else 0 for x in model.predict(X_val)]
#
# # Calculate the confusion matrix
# confusion = confusion_matrix(y_val, predicted_labels)
#
# # Create a heatmap of the confusion matrix
# sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
#
#
# # Convert the model's predictions on the validation set to binary labels (0 or 1)
# predicted_labels = [1 if x >= 0.5 else 0 for x in model2.predict(X_val)]
#
# # Calculate the confusion matrix
# confusion = confusion_matrix(y_val, predicted_labels)
#
# # Create a heatmap of the confusion matrix
# sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
#
# # # Convert the model's predictions on the validation set to binary labels (0 or 1)
# predicted_labels = [1 if x >= 0.5 else 0 for x in model3.predict(X_val)]
#
# # Calculate the confusion matrix
# confusion = confusion_matrix(y_val, predicted_labels)
#
# # Create a heatmap of the confusion matrix
# sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
# #
# predicted_labels = [1 if x >= 0.5 else 0 for x in model4.predict(X_val)]
#
# # Calculate the confusion matrix
# confusion = confusion_matrix(y_val, predicted_labels)
#
# # Create a heatmap of the confusion matrix
# sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
#
# # # # Convert the model's predictions on the validation set to binary labels (0 or 1)
# predicted_labels = [1 if x >= 0.5 else 0 for x in model3.predict(X_val)]
#
# # Calculate the confusion matrix
# confusion = confusion_matrix(y_val, predicted_labels)
#
# # Create a heatmap of the confusion matrix
# sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.show()
#
predicted_labels = [1 if x >= 0.5 else 0 for x in model6.predict(X_val)]

# Calculate the confusion matrix
confusion = confusion_matrix(y_val, predicted_labels)

# Create a heatmap of the confusion matrix
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
# Split your dataset into training and validation sets, then train the model.
# Replace 'train_data' and 'validation_data' with your actual data.


# Step 6: Save the Model
#model.save('image_classifie.h5')
model = keras.models.load_model('image_classifie.h5')
#model2.save('image_classifi.h5')
model2 = keras.models.load_model('image_classifi.h5')
#model3.save('image_classif.h5')
model3 = keras.models.load_model('image_classif.h5')
#model4.save('image_classi.h5')
model4 = keras.models.load_model('image_classi.h5')
#model5.save('image_class.h5')
model5 = keras.models.load_model('image_class.h5')
model6.save('image_clas.h5')
model6 = keras.models.load_model('image_class.h5')
# Directory containing test images
test_directory = 'hvwc23/test/'

# Lists to store test images and filenames
test_images = []
test_filenames = []

# Read and preprocess test images
for filename in os.listdir(test_directory):
    if filename.endswith(".jpg"):
        image_path = os.path.join(test_directory, filename)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (150, 150))
        image = image / 255.0 # Normalize pixel values to [0, 1]
        test_images.append(image)
        test_filenames.append(filename)

test_images = np.array(test_images)

# Use model1 and model2 to make predictions on test images
prediction_model = model.predict(test_images)
prediction_model2 = model2.predict(test_images)
prediction_model3 = model3.predict(test_images)
prediction_model4 = model4.predict(test_images)
prediction_model5 = model5.predict(test_images)
# # Create an array to store ensemble predictions
ensemble_predictions = []
#
count1=0
count2=0
count4=0
count5=0
for pred1, pred2, pred3, pred4, pred5 in zip(prediction_model, prediction_model2, prediction_model3, prediction_model4, prediction_model5):
      if pred1 < 0.5:
          ensemble_predictions.append(0)
          count1=count1+1
      elif pred2 < 0.5:
          ensemble_predictions.append(0)
          count2 = count2 + 1
      elif pred5 > 0.5:
          ensemble_predictions.append(1)
          count5 = count5 + 1
      elif pred3 > 0.5:
            ensemble_predictions.append(1)
            count4 = count4 + 1
      else:
          ensemble_predictions.append(pred4)

predictions = ensemble_predictions
#predictions = prediction_model
# Make predictions for test images
#predictions = model.predict(test_images)

# Load the original CSV file with image IDs and filenames
original_csv = pd.read_csv('hvwc23/test.csv')

# Add the predictions to the DataFrame
original_csv['Prediction'] = predictions
original_csv['Prediction'] = original_csv['Prediction'].apply(lambda x: 1 if x >= 0.5 else 0)


# Save the updated DataFrame to a new CSV file
original_csv.to_csv('updated_file.csv', index=False)
print(count1)
print(count2)
print(count4)
print(count5)
