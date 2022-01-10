import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

datasets = [10,15,20,30]

#Run for each of the 4 datasets
for set in datasets:
  file = "engine21_training_data_"+str(set)+".csv"
  df = pd.read_csv(file)

  # fix random seed for reproducibility
  seed = 7
  numpy.random.seed(seed)

  smooth = df[df[str(set)] == 0]
  knock = df[df[str(set)] == 1]
  knock = knock.sample(n=3000)

  df = knock.append(smooth)

  X = df.iloc[:,0:set]
  Y = df.iloc[:,set]

  # split into 67% for train and 33% for test
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=seed)

  # Train multinomial Naive Bayes model

  mnb = MultinomialNB()
  mnb.fit(X_train, y_train)

  # Make predictions
  y_pred = mnb.predict(X_test)

  # Print accuracy
  print('Model accuracy score with vector length of ' + str(set) + ': {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

  # Print the Confusion Matrix and slice it into four pieces

  cm = confusion_matrix(y_test, y_pred)

  print('Confusion matrix\n\n', cm)

  print('\nTrue Positives(TP) = ', cm[0,0])

  print('\nTrue Negatives(TN) = ', cm[1,1])

  print('\nFalse Positives(FP) = ', cm[0,1])

  print('\nFalse Negatives(FN) = ', cm[1,0])

  # visualize confusion matrix with seaborn heatmap

  cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

  sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
  plt.title('Confusion matrix of vector length ' + str(set))
  plt.show()
