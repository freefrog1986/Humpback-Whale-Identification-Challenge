import numpy as np
import pandas as pd
import keras
import warnings
from os.path import split
from glob import glob
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, Flatten, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras import optimizers

def ImportImage(filename):
    #Image are imported with a resizing and a black and white conversion
    img = Image.open(filename).convert("RGB").resize( (SIZE,SIZE))
    return np.array(img)

class LabelOneHotEncoder():
    def __init__(self):
        self.ohe = OneHotEncoder()
        self.le = LabelEncoder()
    def fit_transform(self, x):
        features = self.le.fit_transform( x)
        return self.ohe.fit_transform( features.reshape(-1,1))
    def transform( self, x):
        return self.ohe.transform( self.la.transform( x.reshape(-1,1)))
    def inverse_tranform( self, x):
        return self.le.inverse_transform( self.ohe.inverse_tranform( x))
    def inverse_labels( self, x):
        return self.le.inverse_transform( x)

def writing_result_to_csv(file_name, the_model):
    with open(working_path+'/'+file_name,"w") as f:
        with warnings.catch_warnings():
            f.write("Image,Id\n")
            warnings.filterwarnings("ignore",category=DeprecationWarning)
        for image in test_images:
            img = ImportImage(image)
            x = img.astype( "float32")
            x = image_gen.standardize(x.reshape(1,SIZE,SIZE,3))
            y = the_model.predict(x.reshape(1,SIZE,SIZE,3))
            predicted_args = np.argsort(y)[0][::-1][:5]
            predicted_tags = lohe.inverse_labels(predicted_args)
            image = split(image)[-1]
            predicted_tags = " ".join( predicted_tags)
            f.write("%s,%s\n" %(image, predicted_tags))
    pass

def recording_score(file_name, the_score):
    with open (working_path+'/'+file_name,'w') as thefile:
        for item in the_score:
            thefile.write("%s\n" % (item))
    pass

if __name__ == '__main__':
    # setting
    working_path = '/mnt/DataLab/Humpback-Whale-Identification-Challenge'
    SIZE = 224

    # training data x
    train_images = glob(working_path + "/inputs/train/*jpg")
    train_img = np.array([ImportImage(img) for img in train_images])
    x = train_img
    x_train = x.reshape((-1,SIZE,SIZE,3)).astype("float32")

    # training data y
    df = pd.read_csv(working_path + "/inputs/train.csv")
    df["Image"] = df["Image"].map(lambda x : working_path + "/inputs/train/" + x)
    ImageToLabelDict = dict(zip(df["Image"], df["Id"]))
    y = list(map(ImageToLabelDict.get, train_images))
    lohe = LabelOneHotEncoder()
    y_cat = lohe.fit_transform(y)
    y_train = y_cat

    # testing data
    test_images = glob(working_path + "/inputs/test/*jpg")

    #constructing class weights
    WeightFunction = lambda x : 1./x**0.75
    ClassLabel2Index = lambda x : lohe.le.inverse_tranform([[x]])
    CountDict = dict(pd.Series(y).value_counts())
    class_weight_dic = {lohe.le.transform([image_name])[0] : WeightFunction(count) for image_name, count in CountDict.items()}
    del CountDict

    #use of an image generator for preprocessing and data augmentation
    image_gen = ImageDataGenerator(
        shear_range=0.15, 
        zoom_range=0.15, 
        rescale=1./255,
        rotation_range=15,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True)
    # fit parameters from data
    image_gen.fit(x_train, augment=True)

    #training parameters
    input_shape = x[0].shape
    batch_size = 128
    num_classes = len(y_train.toarray()[0])
    epochs = 1
    img_input = Input(shape = input_shape)

    try:
        train_data = np.load(open('bottleneck_features_train.npy'))
        print('loading features map done!')
    except Exception as e:
        # create the base pre-trained model
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

        bottleneck_features_train = base_model.predict_generator(image_gen.flow(x_train, y_train.toarray()), x_train.shape[0], verbose=1)
        # save the output as a Numpy array
        np.save(open(working_path + 'bottleneck_features_train.npy', 'wb+'), 
            bottleneck_features_train)

        train_data = np.load(open('bottleneck_features_train.npy'))
        print('loading features map done!')
    finally:
        model = Sequential()
        model.add(Flatten(input_shape=train_data.shape[1:]))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
        print('constructing model done!')
        model.fit(train_data, y_train.toarray(), 
            batch_size = batch_size,
            epochs = epochs,
            verbose = 1,
            class_weight = class_weight_dic)

        model.save_weights((working_path + '/bottleneck_fc_model.h5'))

        # build the VGG16 network
        model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        print('Model loaded.')

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=train_data.shape[1:]))
        top_model.add(Dense(4096, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(num_classes, activation='softmax'))

        # note that it is necessary to start with a fully-trained
        # classifier, including the top classifier,
        # in order to successfully do fine-tuning
        top_model.load_weights(working_path + '\bottleneck_fc_model.h5')

        # add the model on top of the convolutional base
        model.add(top_model)

        # build the VGG16 network
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

        # build a classifier model to put on top of the convolutional model
        top_model = Sequential()
        top_model.add(Flatten(input_shape=train_data.shape[1:]))
        top_model.add(Dense(4096, activation='relu'))
        top_model.add(Dropout(0.7))
        top_model.add(Dense(num_classes, activation='softmax'))

        # note that it is necessary to start with a fully-trained
        # classifier, including the top classifier,
        # in order to successfully do fine-tuning
        top_model.load_weights(working_path + '\bottleneck_fc_model.h5')
        print('Model loaded.')
        # add the model on top of the convolutional base
        model = Model(inputs= base_model.input, outputs= top_model(base_model.output))

        # set the first 25 layers (up to the last conv block)
        # to non-trainable (weights will not be updated)
        for layer in model.layers[:25]:
            layer.trainable = False

        # compile the model with a SGD/momentum optimizer
        # and a very slow learning rate.
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                      metrics=['accuracy'])

        model.fit(x = x_train, 
          y = y_train.toarray(), 
          batch_size = batch_size,
          epochs = 10,
          verbose = 10,
          class_weight = class_weight_dic)
        
        model.save(working_path + 'vgg16_ft_model.h5')  
        print('fine turning model saved!')

        score = model.evaluate(x_train, y_train.toarray(), verbose=0)
        print('Training loss of vgg16 ft model: {0:.4f}\nTraining accuracy of vgg16 ft model:  {1:.4f}'.format(*score))

        #save to file 
        writing_result_to_csv("vgg16_ft_submission.csv", model)
        print('writing to csv done!')