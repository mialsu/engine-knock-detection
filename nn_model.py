
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy
from sklearn.model_selection import cross_val_score

datasets = [10,15,20,30]

# Run for each of the 4 datasets
for set in datasets:
  file = "engine21_training_data_"+str(set)+".csv"
  df = pd.read_csv(file)

  # fix random seed for reproducibility
  seed = 7
  numpy.random.seed(seed)

  smooth = df[df[str(set)] == 0]
  knock = df[df[str(set)] == 1]

  #Take sample of 3000 pressure variance vectors from knocking data
  knock = knock.sample(n=3000)

  df = knock.append(smooth)

  X = df.iloc[:,0:set]
  Y = df.iloc[:,set]

  # split into 67% for train and 33% for test
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

  # define the keras model
  model = Sequential()
  model.add(Dense(set+20, input_dim=set, activation='relu'))
  model.add(Dense(set+10, activation='relu'))
  model.add(Dense(1, activation='sigmoid'))
  # compile the keras model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  # fit the keras model on the dataset
  history = model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=300, batch_size=10)

  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('Model accuracy for ' + str(set) + ' vector length')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show() 
  plt.savefig('accuracy_' + str(set) + '_vector_length.png') 

  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model loss for ' + str(set) + ' vector length')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  plt.savefig('loss_' + str(set) + '_vector_length.png') 