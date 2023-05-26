import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\Training\reviews.csv")
# df1 = df.iloc[:, [0, 1, 3, 2]].copy()
# df1['label'].replace(['CG', 'OR'], [0, 1], inplace=True)
#
# xfeatures = df['text_']
# yfeatures = df['label']
# x_train, x_test, y_train, y_test = train_test_split(xfeatures, yfeatures, test_size=0.2)
#
# pipe = Pipeline([('tfidf', TfidfVectorizer()), ('lr', LogisticRegression(max_iter=5000))])
#
# parameters = {'tfidf__max_df': [0.5, 1.0], 'tfidf__ngram_range': [(1, 1), (1, 2)],
#               'lr__C': [0.1, 1, 10], 'lr__penalty': ['l1', 'l2']}
#
# grid_search = GridSearchCV(pipe, parameters, cv=5, n_jobs=-1, verbose=2)
# grid_search.fit(x_train, y_train)
#
# print("Best parameters: ", grid_search.best_params_)
#
# y_predict = grid_search.predict(x_test)
#
# cr = classification_report(y_test,y_predict)
# print(cr)
# acc = accuracy_score(y_test,y_predict)
# print(acc)
#











# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import classification_report, accuracy_score
#
# # Load the dataset
# df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\Training\reviews.csv")
#
# # Select relevant columns and convert label values to numerical
# df1 = df.iloc[:, [0, 1, 3, 2]]
# df1['label'].replace(['CG', 'OR'], [0, 1], inplace=True)
#
# # Split the data into training and testing sets
# xfeatures = df['text_']
# yfeatures = df['label']
# x_train, x_test, y_train, y_test = train_test_split(xfeatures, yfeatures, test_size=0.2)
#
# # Vectorize the text data using TF-IDF
# vectorization = TfidfVectorizer()
# transformed_output = vectorization.fit_transform(df['text_'])
# feature_names = vectorization.get_feature_names_out()
#
# # Convert category values to numerical and join with the vectorized text data
# df1['category'].replace(
#     ['Home_and_Kitchen_5', 'Sports_and_Outdoors_5', 'Electronics_5', 'Movies_and_TV_5', 'Tools_and_Home_Improvement_5',
#      'Pet_Supplies_5', 'Kindle_Store_5', 'Books_5', 'Toys_and_Games_5',
#      'Clothing_Shoes_and_Jewelry_5'],
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
# df2 = pd.DataFrame(transformed_output.toarray(), columns=vectorization.vocabulary_)
# df2['Mean'] = df2.mean(axis=1)
# extracted_col = df2["Mean"]
# df3 = df1.join(extracted_col)
#
# # Drop the text column and split into features and labels
# df4 = df3.drop(['text_'], axis=1)
# x = df4[['category', 'rating', 'Mean']]
# y = df4[['label']]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#
# # Train a Naive Bayes model and make predictions on the testing set
# classifier = GaussianNB()
# classifier.fit(x_train, y_train)
# y_predict = classifier.predict(x_test)
#
# # Print classification report and accuracy score
# cr = classification_report(y_test, y_predict)
# print(cr)
# acc = accuracy_score(y_test, y_predict)
# print(acc)
#
#
#
#
#
#











# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# import matplotlib.pyplot as mtp
# from sklearn.naive_bayes import GaussianNB
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
#
# df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\Training\reviews.csv")
# df1 = df.iloc[:, [0, 1, 3, 2]]
# df1['label'].replace(['CG', 'OR'], [0, 1], inplace=True)
# xfeatures = df['text_']
# yfeatures = df['label']
# x_train, x_test, y_train, y_test = train_test_split(xfeatures, yfeatures, test_size=0.2)
# vectorization = TfidfVectorizer()
# transformed_output = vectorization.fit_transform(df['text_'])
# feature_names = vectorization.get_feature_names_out()
# df1['category'].replace(
#     ['Home_and_Kitchen_5', 'Sports_and_Outdoors_5', 'Electronics_5', 'Movies_and_TV_5', 'Tools_and_Home_Improvement_5',
#      'Pet_Supplies_5', 'Kindle_Store_5', 'Books_5', 'Toys_and_Games_5',
#      'Clothing_Shoes_and_Jewelry_5'],
#     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
# df2 = pd.DataFrame(transformed_output.toarray(), columns=vectorization.vocabulary_)
# df2['Mean'] = df2.mean(axis=1)
# extracted_col = df2["Mean"]
# df3 = df1.join(extracted_col)
#
# df4=df3.drop(['text_'], axis=1)
# x=df4[['category','rating','Mean']]
# y=df4[['label']]
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# classifier = GaussianNB()
# classifier.fit(x_train, y_train)
# y_predict = classifier.predict(x_test)
# cr = classification_report(y_test,y_predict)
# print(cr)
# acc = accuracy_score(y_test,y_predict)
# print(acc)


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pickle

# Load the training data
df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\Training\reviews.csv")
df1 = df.iloc[:, [0, 1, 3, 2]]
df1['label'].replace(['CG', 'OR'], [0, 1], inplace=True)
xfeatures = df['text_']
yfeatures = df['label']

# Perform text vectorization
vectorization = TfidfVectorizer()
transformed_output = vectorization.fit_transform(df['text_'])
df2 = pd.DataFrame(transformed_output.toarray(), columns=vectorization.get_feature_names_out())
df2['Mean'] = df2.mean(axis=1)
extracted_col = df2["Mean"]
df3 = df1.join(extracted_col)
df4 = df3.drop(['text_'], axis=1)
x_train = df4[['category', 'rating', 'Mean']]
y_train = df4[['label']]

# Train the GaussianNB classifier
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Save the trained classifier as a .sav file
filename = 'classifier_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Load the saved model from .sav file
loaded_model = pickle.load(open(filename, 'rb'))

# Load the Flipkart generated reviews from a CSV file
flipkart_reviews_df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\Generated reviews\flipkart_reviews.csv")  # Update with the actual path to the CSV file

# Perform the necessary preprocessing on the Flipkart generated reviews
# Extract features using the same vectorizer used during training
transformed_reviews = vectorization.transform(flipkart_reviews_df['text_'])
df_flipkart = pd.DataFrame(transformed_reviews.toarray(), columns=vectorization.get_feature_names_out())
df_flipkart['Mean'] = df_flipkart.mean(axis=1)
features_flipkart = df_flipkart[['category', 'rating', 'Mean']]

# Make predictions using the loaded model
predictions = loaded_model.predict(features_flipkart)

# Add the predictions to the Flipkart reviews DataFrame
flipkart_reviews_df['predicted_label'] = predictions

# Print or manipulate the Flipkart reviews DataFrame with predicted labels as needed
print(flipkart_reviews_df)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pickle

# Load the training data
df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\Training\reviews.csv")
df1 = df.iloc[:, [0, 1, 3, 2]]
df1['label'].replace(['CG', 'OR'], [0, 1], inplace=True)
xfeatures = df['text_']
yfeatures = df['label']

x_train, x_test, y_train, y_test = train_test_split(xfeatures, yfeatures, test_size=0.2)
# Perform text vectorization
vectorization = TfidfVectorizer()
transformed_output = vectorization.fit_transform(df['text_'])
feature_names = vectorization.get_feature_names_out()
df1['category'].replace(
    ['Home_and_Kitchen_5', 'Sports_and_Outdoors_5', 'Electronics_5', 'Movies_and_TV_5', 'Tools_and_Home_Improvement_5',
     'Pet_Supplies_5', 'Kindle_Store_5', 'Books_5', 'Toys_and_Games_5',
     'Clothing_Shoes_and_Jewelry_5'],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
df2 = pd.DataFrame(transformed_output.toarray(), columns=vectorization.get_feature_names_out())
df2['Mean'] = df2.mean(axis=1)
extracted_col = df2["Mean"]
df3 = df1.join(extracted_col)
df4 = df3.drop(['text_'], axis=1)
x = df4[['category', 'rating', 'Mean']]
y = df4[['label']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
# Train the GaussianNB classifier
classifier = GaussianNB()
classifier.fit(x , y)

# Save the trained classifier as a .sav file
filename = 'adaptive_model.sav'
pickle.dump(classifier, open(filename, 'wb'))

# Load the saved model from .sav file
loaded_model = pickle.load(open(filename, 'rb'))

# Load the Flipkart generated reviews from a CSV file
flipkart_reviews_df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\Generated reviews\flipkart_reviews.csv")  # Update with the actual path to the CSV file

# Perform the necessary preprocessing on the Flipkart generated reviews
# Extract features using the same vectorizer used during training
transformed_reviews = vectorization.transform(flipkart_reviews_df['text_'])
df_flipkart = pd.DataFrame(transformed_reviews.toarray(), columns=vectorization.get_feature_names_out())
df_flipkart['Mean'] = df_flipkart.mean(axis=1)
features_flipkart = df_flipkart[['category', 'rating', 'Mean']]

# Make predictions using the loaded model
predictions = loaded_model.predict(features_flipkart)

# Add the predictions to the Flipkart reviews DataFrame
flipkart_reviews_df['predicted_label'] = predictions

# Print or manipulate the Flipkart reviews DataFrame with predicted labels as needed
print(flipkart_reviews_df)

