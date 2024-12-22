
import numpy as numpy
import matplotlib.pyplot as pyplot
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import pickle
import os
import pandas as pandas
import random
from keras.preprocessing.image import ImageGenerator

############Parameters#############

path = "myData"
labelFile = 'labels.csv'
batch_size_value = 10 #50
steps_per_epoch_value = 500 #2000
epochs_value = 5 #30
imageDimensions = (32, 32, 3) 
testRatio = 0.2
validationRatio = 0.2

#########Importing images

count = 0
images = []
classNr = []
myList = os.listdir(path)
print("Total classes Detected", len(myList))
nrOfClasses = len(myList)
print("importing classes")

for x in range(o, len(myList)):
    myPicList = os.listdir(path+"/"+str(count))
    for y in myPicList:
        currentImage = cv2.imread(path+"/"+str(count)+"/"+y)
        images.append(currentImage)
        classNr.append(count)
    print(count, end="")
    count += 1
print(" ")
images = numpy.array(images)
classNr = numpy.array(classNr)

X_train, X_test, y_train, y_test = train_test_split(images, classNr, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size = validationRatio)

# X_train = array of images to train
# y_train = corresponding class id

##############check if number of images matches to number of labels for each dataset

print("Data Shapes")
print("Train", end = ""); print(X_train.shape, y_train.shape)
print("Validation", end = ""); print(X_test.shape, y_test.shape)
print("Test", end = ""): print(X_test.shape, y_test.shape)

assert(X_train.shape[0] == y_train.shape[0]), "Nr of images in is != nr of labels in training set"
assert(X_validation.shape[0] == y_validation.shape[0]), "Nr of images != nr of labels in validation set"
assert(X_test.shape[0] == y_test.shape[0]), "Nr of images != nr of labels in test set"
assert(X_train.shape[1:] == (imageDimensions)), "The dimensions of the Training images are wrong"
assert(X_validation.shape[1:]== (imageDimensions)), "Dimensions of the validation images are wrong"
assert(X_test.shape[1:] == (imageDimensions)), "Dimensions of test images are wrong"

#####read CSV

data = pandas.read_csv(labelFile)
print("data shape ", data.shape, type(data))

###############display sample images of classes

num_of_samples = []
cols = 5
num_classes = nrOfClasses
fig, axs = pyplot.subplots(nrows = num_classes, ncols = cols, figsize=(5, 300))
fig.tight_layout()

for i in range(cols):
    for j,row in data.iterrows():
        x_selected = X_train[y_train == j]
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap = pyplot.get_cmp("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + row["Name"])
            num_of_samples.append(len(x_selected))

################display a bar chart showing nr of samples for each category
            
print(num_of_samples)
pyplot.figure(figsize=(12, 4))
pyplot.bar(range(0, num_classes), num_of_samples)
pyplot.title("Distribution of training dataset")
pyplot.xlabel("Class nr")
pyplot.ylabel("Nr of images")
pyplot.show()

############# preprocessing images

def grayScale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img = cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayScale(img) #convert to grayscale
    img = equalize(img)  #standarize lightning in image
    img = img/255        #normalize values between 0 and 1, instead of 0-255
    return img

X_train = numpy.array(list(map(preprocessing, X_train)))
X_validation = numpy.array(list(map(preprocessing, X_validation)))
X_test = numpy.array(list(map(preprocessing, X_test)))

#####add depth of 1

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shaoe[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shaoe[1], X_test.shaoe[2], 1)

######## Augemntation of images => more generic

dataGen = ImageDataGenerator(width_shift_range = 0.1, #10% if more than 1 then it refers to nr of pixels (10)
                             height_shift_range = 0.1, 
                             zoom_range = 0.2, # means +- 0,2 i.e 0.8-1.2
                             shear_range = 0.1, # magnitude of shear angle
                             rotation_range = 10) #degrees

dataGen.fit(X_train)
batches = dataGen.flow(X_train, y_train, batch_size = 20) # requesting data generator to generate images batch size = nr images created each time its called
X_batch, y_batch = next(batches)

#show augmented image samples

fig, axs = pyplot.subplot(1, 15, figsize=(20, 5))
fig.tight_layout()

for i in range(15):
    axs[i].imshow(X_batch[i].reshape(imageDimensions[0], imageDimensions[1]))
    axs[i].axis('off')
pyplot.show()

y_train = to_categorical(y_train, nrOfClasses)
y_validation = to_categorical(y_validation, nrOfClasses)
y_test = to_categorical(y_test, nrOfClasses)

def model():
    nr_of_filters = 60
    size_of_Filter=(5,5) #This is the kernel that move around the image to get the features
    size_of_Filter2 = (3,3)
    size_of_pool = (2,2)
    nr_of_Nodes = 100 # 500
    model = Sequential()
    model.add((Conv2D(nr_of_filters, size_of_Filter, input_shape = (imageDimensions[0], imageDimensions[1], 1), activation = 'relu')))
    model.add((Conv2D(nr_of_filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool)) #does not effect the depth/nr of filters

    model.add((Conv2D(nr_of_filters//2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(nr_of_filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool)) #does not effect the depth/nr of filters
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(nr_of_Nodes, activation = 'relu'))
    model.add(Dropout(0.5)) #input nodes to drop with each update, 1 all, 0 none
    model.add(Dense(nrOfClasses, activation = 'softmax')) #output layer

    model.compile(Adam(lr=0.001), loss='categorical coressentropy', metrics=['accuracy'])
    return model

mymodel = model()
print(mymodel.summary())
history = mymodel.fit_generator(dataGen.flow(X_train, y_train, 
                                             batch_size = batch_size_value), 
                                steps_per_epoch = steps_per_epoch_value,
                                epochs = epochs_value,
                                validation_data =(X_validation, y_validation),
                                shuffle = 1)

pyplot.figure(1)
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['value_loss'])
pyplot.legend(['training', 'validation'])
pyplot.title('Loss')
pyplot.xlabel('epoch')
pyplot.figure(2)
pyplot.plot(history.history['accuray'])
pyplot.plot(history.history['val_accuracy'])
pyplot.legend(['training','validation'])
pyplot.title('Accuracy')
pyplot.xlabel('epoch')
pyplot.show()

#### evaulating with test images
score = mymodel.evaluate(X_test,y_test,verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])

#### Save trained model
pickle_out = open("model_trained.p", "wb")
pickle.dump(mymodel, pickle_out)
pickle_out.close()