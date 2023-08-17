#Importing necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Reading customer review data from text file into a pandas dataframe
df = pd.read_csv('customerReviews.txt', header=None, names=['Review', 'Sentiment'], error_bad_lines=False)

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['Review'], df['Sentiment'], random_state=0)

# Creating a CountVectorizer and fitting it to the training data
vectorize_bayes = CountVectorizer().fit(X_train)

# Transforming the training data using the CountVectorizer
X_train_vectorized = vectorize_bayes.transform(X_train)

# Creating a Multinomial Naive Bayes classifier and fitting it to the vectorized training data
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Transforming the test data using the CountVectorizer
X_test_vectorized = vectorize_bayes.transform(X_test)

# Predicting sentiments of the test data
y_pred = clf.predict(X_test_vectorized)

# Calculating the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Sample reviews to predict their sentiments
examples = ['Their customer support was rude and unprofessional',
            'I am very impressed with the quality of this product',
            'The item I received was defective and did not work',
            'The pricing is reasonable for the features provided']

# Transforming the sample reviews using the CountVectorizer
examples_vectorized = vectorize_bayes.transform(examples)

# Predicting sentiments of the sample reviews
predictions = clf.predict(examples_vectorized)

# Printing the sample reviews and their predicted sentiments
for review, prediction in zip(examples, predictions):
    print(review, '->', prediction)
