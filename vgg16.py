import numpy as np
import pandas as pd
import keras
import warnings
from glob import glob
from PIL import Image
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from subprocess import check_output
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from os.path import split
from keras.layers import Input
from keras.utils.data_utils import get_file
from keras.layers import GlobalMaxPooling2D
from keras.models import Model

def ImportImage( filename):
    #Image are imported with a resizing and a black and white conversion
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

def cnn_model():
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
    return model

def vgg16_model():
    # 编写网络结构，prototxt
    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(num_classes, activation='softmax', name='predictions')(x) #改变了输出数量

    # Create model.
    model = Model(img_input, x, name='vgg16')

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    model.summary()
    model.fit_generator(image_gen.flow(x_train, y_train.toarray(), batch_size=batch_size),
                        steps_per_epoch = x_train.shape[0]//batch_size,
                        epochs = epochs,
                        verbose=1,
                        class_weight = class_weight_dic)
    return model

def writing_result_to_csv(file_name, the_model):
    with open(working_path+'/'+file_name,"w") as f:
        with warnings.catch_warnings():
            f.write("Image,Id\n")
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            for image in test_images:
                img = ImportImage( image)
                x = img.astype( "float32")
                x = image_gen.standardize(x.reshape(1,SIZE,SIZE))
                if str(vgg16_model)[23:28] == 'Model':
                    y = the_model.predict(x.reshape(1,SIZE,SIZE,1))
                else:
                    y = the_model.predict_proba(x.reshape(1,SIZE,SIZE,1))
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
    working_path = '/mnt/DataLab/Humpback-Whale-Identification-Challenge'
    train_images = glob(working_path + "/inputs/train/*jpg")
    test_images = glob(working_path + "/inputs/test/*jpg")
    df = pd.read_csv(working_path + "/inputs/train.csv")

    df["Image"] = df["Image"].map(lambda x : working_path + "/inputs/train/"+x)
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
    CountDict = dict( df["Id"].value_counts())
    class_weight_dic = { lohe.le.transform( [image_name])[0] : WeightFunction(count) for image_name, count in CountDict.items()}
    del CountDict

    #plotting training images from training set after resizing and BW conversion
    #plotImages( x)

    #use of an image generator for preprocessing and data augmentation
    x = x.reshape( (-1,SIZE,SIZE,1))
    input_shape = x[0].shape
    x_train = x.astype("float32")
    y_train = y_cat

    image_gen = ImageDataGenerator(featurewise_center = True, 
                                   samplewise_center = False, 
                                   featurewise_std_normalization=True, 
                                   samplewise_std_normalization=False, 
                                   zca_whitening = True, 
                                   zca_epsilon=1e-06, 
                                   rotation_range = 10, 
                                   width_shift_range=0.1, 
                                   height_shift_range=0.1, 
                                   shear_range=0.1, 
                                   zoom_range=0.1, 
                                   channel_shift_range=0.0, 
                                   fill_mode='nearest', 
                                   cval=0.0, 
                                   horizontal_flip=False, 
                                   vertical_flip=False, 
                                   rescale=1./255, 
                                   preprocessing_function=None, 
                                   data_format = 'channels_last')
    # fit parameters from data
    image_gen.fit(x_train, augment=True)

    #training parameters
    batch_size = 128
    num_classes = len(y_cat.toarray()[0])
    epochs = 100
    img_input = Input(shape = input_shape)
    try:
        #model training
        #cnn_model = cnn_model()
        vgg16_model = vgg16_model()

        # evaluate
        #cnn_score = cnn_model.evaluate(x_train, y_train, verbose=0)
        #print('Training loss of cnn model: {0:.4f}\nTraining accuracy of cnn model:  {1:.4f}'.format(*cnn_score))
        vgg16_score = vgg16_model.evaluate(x_train, y_train, verbose=0)
        print('Training loss of vgg16 model: {0:.4f}\nTraining accuracy of vgg16 model:  {1:.4f}'.format(*vgg16_score))

        #save to file 
        #writing_result_to_csv("cnn_submission.csv", cnn_model)
        writing_result_to_csv("vgg16_submission.csv", vgg16_model)

        #save score to file
        #recording_score("cnn_score.txt", cnn_score)
        recording_score("vgg16_score.txt", vgg16_score)
    except Exception as e:
        with open('exception.txt','w') as f:
            f.write(str(e))
        raise
    
    