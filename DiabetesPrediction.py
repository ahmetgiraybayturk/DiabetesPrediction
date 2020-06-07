#Adding required libraries
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#Load the dataset
file = pd.read_csv('stats.csv')
#First 20 rows of data
file.head(20)

#Checking for duplicates and removing them
file.drop_duplicates(inplace = True)

#Show the shape (number of rows & columns)
file.shape

#Show the number of missing (NAN, NaN, na) data for each column
file.isnull().sum()

#Convert the data into an array
dataset = file.values
dataset

# Get all of the rows from the first eight columns of the dataset
X = dataset[:,0:8] 
# Get all of the rows from the last column
y = dataset[:,8] 

#Process the feature data set to contain values between 0 and 1 inclusive
#by using the min-max scaler method, and print the values.
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_scale

#Split the data into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X_scale, y, test_size=0.2, random_state = 4)

#Building the ANN
#First layer has 12 neurons and use the ReLu activation func.
#Second layer has 15 neurons and use the ReLu activation func.
#Third layer uses 1 neuron and the sigmoid activation func.
model = Sequential([
    Dense(12, activation='relu', input_shape=( 8 ,)),
    Dense(15, activation='relu'),
    Dense(1, activation='sigmoid')])

#Binary_crossentropy loss function binary classification to measure how well the model did on training
#Stochastic Gradient Descent ‘sgd’ optimizer to improve upon the loss
#To measure the accuracy of the model ‘accuracy’ added to the metrics
model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['acc'])

#Train the model by using the fit method on the training data, and train it in batch sizes of , with 1000 epochs.
#Give the model validation data to see how well the model is performing by splitting the training data into 20% validation.
#Batch:Total number of training examples present in a single batch
#Epoch:The number of iterations when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
hist = model.fit(X_train, y_train,batch_size=45, epochs=1000, validation_split=0.2)


#Visualizing the training loss and the validation loss to see if the model is overfitting
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

#Visualizing the training accuracy and the validation accuracy to see if the model is overfitting
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()

#Evaluating the model on the training data set
from sklearn.metrics import accuracy_score
pred = model.predict(X_train)
pred  = [1 if y>=0.5 else 0 for y in pred] #Threshold
print('Accuracy: ', accuracy_score(y_train,pred))















