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
import datetime

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
    model.fit_generator(image_gen.flow(x_train, y_train.toarray(), batch_size=batch_size),
              steps_per_epoch=  x_train.shape[0]//batch_size,
              epochs=epochs,
              verbose=1,
              class_weight=class_weight_dic)
    return model

def writing_result_to_csv(file_name, result):
    with open(working_path+'/'+file_name,"w") as f:
        with warnings.catch_warnings():
            f.write("Image,Id\n")
            warnings.filterwarnings("ignore",category=DeprecationWarning)
            for image,preds in result.items():
                image = split(image)[-1]
                predicted_tags = []
                for i in map(lambda x:x[0],sorted(preds.items(), key=lambda a:a[1], reverse=True)[:5]):
                    predicted_tags.append(i)
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
    #working_path = '/Users/freefrog/Studing/DataScience/Machine-Learning-Coursera/inputs' # working locally
    train_images = glob(working_path + "/inputs/train/*jpg")
    test_images = glob(working_path + "/inputs/test/*jpg")
    df = pd.read_csv(working_path + "/inputs/train.csv")

    df["Image"] = df["Image"].map(lambda x : working_path + "/inputs/train/"+x)
    ImageToLabelDict = dict( zip( df["Image"], df["Id"]))

    SIZE = 64

    train_img = np.array([ImportImage( img) for img in train_images])

    y_list = list(map(ImageToLabelDict.get, train_images))

    # whale_dict keeps indicies of each whale id  
    whale_dict = {} 
    for whale_id in set(y_list):
        ix = np.isin(y_list, whale_id)
        indices = np.where(ix)
        whale_dict[whale_id] = indices

    #function for class weights
    WeightFunction = lambda x : 1./x**0.75
    ClassLabel2Index = lambda x : lohe.le.inverse_tranform([[x]])

    # image_gen for training data
    image_gen = ImageDataGenerator(featurewise_center = True, 
                                   samplewise_center = False, 
                                   samplewise_std_normalization=False, 
                                   zca_whitening = True, 
                                   zca_epsilon=1e-06, 
                                   rotation_range = 10, 
                                   width_shift_range=0.15, 
                                   height_shift_range=0.15, 
                                   shear_range=0.15, 
                                   zoom_range=0.15, 
                                   channel_shift_range=0.0, 
                                   fill_mode='nearest', 
                                   cval=0.0, 
                                   horizontal_flip=False, 
                                   vertical_flip=False, 
                                   rescale=1./255, 
                                   preprocessing_function=None, 
                                   data_format = 'channels_last')
    #training parameters
    batch_size = 128
    num_classes = 2
    epochs = 10
    result = {}

    # loop over to get all cnn models
    try:
        loop_conter = 0
        for whale,indexes in whale_dict.items():
            starttime = datetime.datetime.now() #used to compute running time
            # Training data
            x1 = train_img[whale_dict[whale][0]]#positive samples

            index_x = np.array(range(len(y_list)))
            mask= np.isin(index_x, whale_dict[whale][0], invert=True)
            index_x = index_x[mask]#index without positive samples

            if len(whale_dict[whale][0]) < 500:
                random_index_x = np.random.choice(index_x, 500) 
            else :
                random_index_x = np.random.choice(index_x, len(whale_dict[whale][0]))

            x2 = train_img[random_index_x]# take random negtive samples
            
            x_train = np.vstack((x1,x2))# stack x1 and x2 to make the training dataset
            
            x_train = x_train.reshape( (-1,SIZE,SIZE,1))
            input_shape = x_train[0].shape
            x_train = x_train.astype("float32")
            
            # Training labels
            two_class_list = [whale if x < len(whale_dict[whale][0]) else 'NE' for x in range(x_train.shape[0])]

            lohe = LabelOneHotEncoder()
            y_cat = lohe.fit_transform(two_class_list)# convert labels to onehot vectoer
            y_train = y_cat
            
            # Image generator for preprocessing and data augmentation
            image_gen.fit(x_train, augment=True)
            
            # Constructing class weights
            CountDict = dict(pd.Series(two_class_list).value_counts())
            class_weight_dic = {lohe.le.transform([image_name])[0]: WeightFunction(count) for image_name, count in CountDict.items()}
            del CountDict

            #model training
            the_cnn_model = cnn_model()
            # evaluate
            cnn_score = the_cnn_model.evaluate(x_train, y_train, verbose=0)
            print('Training loss of cnn model: {0:.4f}\nTraining accuracy of cnn model:  {1:.4f}'.format(*cnn_score))
            
            for image in test_images[:10]:
                img = ImportImage(image)
                x = img.astype( "float32")
                x = image_gen.standardize(x.reshape(1,SIZE,SIZE))
                y = the_cnn_model.predict_proba(x.reshape(1,SIZE,SIZE,1)) 
                try:
                    result[image][whale] = y[0][0]
                except KeyError:
                    result[image] = {whale:y[0][0]} 

            loop_conter +=1
            endtime = datetime.datetime.now()
            remaining_time = (endtime - starttime)*((len(whale_dict)) - loop_conter)
            print('remaining time is %s' % str(remaining_time))
        # write results to file 
        writing_result_to_csv('combined_cnn_model', result)
        del result
        del the_cnn_model
        del lohe
    except Exception as e:
        with open('exception.txt','w') as f:
            f.write(str(e))
        raise
