from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.preprocessing.image import *
from tensorflow import keras as keras
import numpy as np
import cv2
from  PIL import Image
import matplotlib.pyplot as plt
#from  model import inception
def train():
    '''
    inception_builder = inception.Inceptionv3_builder()
    model = inception_builder.build_inception()
    
    #model.summary()
    #keras.utils.plot_model(model, 'small_resnet_model.png', show_shapes=True)
    '''
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    test_datagen=ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(299, 299),
        batch_size=32,
    )

    validation_generator = test_datagen.flow_from_directory(
        'valiation',
        target_size=(299, 299),
        batch_size=32,
    )
    test_generator = test_datagen.flow_from_directory("test2", (224, 224), shuffle=False,
                                             batch_size=16, class_mode=None)
    base_model = InceptionV3(input_tensor=Input(shape=(299, 299, 3)),weights='imagenet', include_top=False)
    #base_model.summary()
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(2, activation='softmax')(x)
    model = Model(base_model.input, predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,  # 2000 images=batch_szie*steps
        epochs=10,
        validation_data=validation_generator,
        validation_steps=50,  # 1000=20*50
        verbose=2)

    for i, layer in enumerate(base_model.layers):
        print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 172 layers and unfreeze the rest:
    for layer in model.layers[:172]:
        layer.trainable = False
    for layer in model.layers[172:]:
        layer.trainable = True
    model.compile(keras.optimizers.SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    history_ft = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=50,
        )
    model.save("weights/inceptionv3.h5")
    y_pred=model.predict(test_generator)
    y_pred = y_pred.clip(min=0.005, max=0.995)
    import pandas as pd
    df = pd.read_csv("sample_submission.csv")
    gen = ImageDataGenerator()
    test_generator = gen.flow_from_directory("test", (224, 224), shuffle=False,
                                             batch_size=16, class_mode=None)
    for i, fname in enumerate(test_generator.filenames):
        index = int(fname[fname.rfind('/') + 1:fname.rfind('.')])
        df.set_value(index - 1, 'label', y_pred[i])

    df.to_csv('pred.csv', index=None)
    df.head(10)

if __name__ == "__main__":
    train()
