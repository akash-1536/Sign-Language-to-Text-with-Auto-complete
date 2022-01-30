# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense , Dropout
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
sz = 310
# Step 1 - Building the CNN

# Initializing the CNN
classifier = Sequential()

classifier.add(Convolution2D(32, (3, 3), input_shape=(sz, sz, 1), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Convolution2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the layers
classifier.add(Flatten())

# Adding a fully connected layer
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dropout(0.30))
classifier.add(Dense(units=27, activation='softmax')) # softmax for more than 2





# Compiling the CNN
classifier.compile(optimizer='adam',
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
# categorical_crossentropy for more than 2


# Step 2 - Preparing the train/test data and training the model
classifier.summary()
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

num_samples=12845
batch_size=5
validaton_samples=4268

training_set = train_datagen.flow_from_directory('dataset_massey_univ/data/preprocessed/train',
                                                 target_size=(sz, sz),
                                                 batch_size=batch_size,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')


test_set = train_datagen.flow_from_directory('dataset_massey_univ/data/preprocessed/test',
                                            target_size=(sz , sz),
                                            batch_size=batch_size,
                                            color_mode='grayscale',
                                            class_mode='categorical')

classifier.fit_generator(
        training_set,
        #steps_per_epoch=1284, # No of images in training set
        steps_per_epoch=num_samples // batch_size,
        epochs=5,
        validation_data=test_set,
        #validation_steps=426)# No of images in validation set
        validation_steps=validaton_samples//batch_size
)


# scores=classifier.evaluate_generator(test_set,1284)
scores=classifier.evaluate_generator(test_set,validaton_samples//batch_size)
print("Accuracy=",scores[1])

# Saving the model
model_json = classifier.to_json()
with open("model/model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model/model-bw.h5')
print('Weights saved')

