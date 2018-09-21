# This script is from https://www.kaggle.com/gimunu/data-augmentation-with-keras-into-cnn

import numpy as np
import pandas as pd
from glob import glob
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

#image are imported with a resizing and a black and white conversion
def ImportImage( filename):
    img = Image.open(filename).convert("LA").resize( (SIZE,SIZE))
    return np.array(img)[:,:,0]

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

def plotImages( images_arr, n_images=4):
    fig, axes = plt.subplots(n_images, n_images, figsize=(12,12))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        if img.ndim != 2:
            img = img.reshape( (SIZE,SIZE))
        ax.imshow( img, cmap="Greys_r")
        ax.set_xticks(())
        ax.set_yticks(())
    plt.tight_layout()

if __name__ == '__main__':
    working_path = '/mnt/DataLab/Humpback-Whale-Identification-Challenge'
    train_images = glob(working_path + "/inputs/train/*jpg")
    test_images = glob(working_path + "/inputs/test/*jpg")
    df = pd.read_csv(working_path + "/inputs/train.csv")

    df["Image"] = df["Image"].map( lambda x :working_path +  "/inputs/train/"+x)
    ImageToLabelDict = dict( zip( df["Image"], df["Id"]))

    SIZE = 64

    train_img = np.array([ImportImage( img) for img in train_images])
    x = train_img

    y = list(map(ImageToLabelDict.get, train_images))
    lohe = LabelOneHotEncoder()
    y_cat = lohe.fit_transform(y)

    #constructing class weights
    WeightFunction = lambda x : 1./x**0.75
    ClassLabel2Index = lambda x : lohe.le.inverse_tranform( [[x]])
    CountDict = dict(pd.Series(y).value_counts())
    class_weight_dic = { lohe.le.transform( [image_name])[0] : WeightFunction(count) for image_name, count in CountDict.items()}
    del CountDict
    
    #plotting training images from training set after resizing and BW conversion
    #plotImages( x)

    #use of an image generator for preprocessing and data augmentation
    x = x.reshape( (-1,SIZE,SIZE,1))
    input_shape = x[0].shape
    x_train = x.astype("float32")
    y_train = y_cat

    image_gen = ImageDataGenerator(
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        rescale=1./255,
        rotation_range=15,
        width_shift_range=.15,
        height_shift_range=.15,
        horizontal_flip=True)

    #training the image preprocessing
    image_gen.fit(x_train, augment=True)

    #visualization of some images out of the preprocessing
    #augmented_images, _ = next( image_gen.flow( x_train, y_train.toarray(), batch_size=4*4))
    #plotImages( augmented_images)
    batch_size = 128
    num_classes = len(y_cat.toarray()[0])
    epochs = 9

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')

    model = Sequential()
    model.add(Conv2D(48, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(48, (3, 3), activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.33))
    model.add(Flatten())
    model.add(Dense(36, activation='sigmoid'))
    model.add(Dropout(0.33))
    model.add(Dense(36, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    model.fit_generator(image_gen.flow(x_train, y_train.toarray(), batch_size=batch_size),
              steps_per_epoch=  x_train.shape[0]//batch_size,
              epochs=epochs,
              verbose=1,
              class_weight=class_weight_dic)

    #K.clear_session()
    score = model.evaluate(x_train, y_train, verbose=0)
    print('Training loss of model: {0:.4f}\nTraining accuracy of model:  {1:.4f}'.format(*score))
    print('training done')

    import warnings
    from os.path import split

    with open(working_path + "sample_submission.csv","w") as f:
        with warnings.catch_warnings():
            f.write("Image,Id\n")
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            for image in test_images:
                img = ImportImage( image)
                x = img.astype( "float32")
                #applying preprocessing to test images
                x = image_gen.standardize( x.reshape(1,SIZE,SIZE))
                
                #K.clear_session()
                y = model.predict_proba(x.reshape(1,SIZE,SIZE,1))
                predicted_args = np.argsort(y)[0][::-1][:5]
                predicted_tags = lohe.inverse_labels( predicted_args)
                image = split(image)[-1]
                predicted_tags = " ".join( predicted_tags)
                f.write("%s,%s\n" %(image, predicted_tags))
    print('all done')