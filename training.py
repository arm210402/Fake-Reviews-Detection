import os
import spacy
import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from skmultiflow.drift_detection import DDM, EDDM, ADWIN

#
# def training():
#     st.header("Training")
#     folder_path = 'C:/Users/Akshay/Desktop/flipkart reviews/Training'
#     files = os.listdir(folder_path)
#     uploaded_file = st.selectbox('Select a file', files)
#     print(uploaded_file)
#     # Create a dropdown menu to select a classification model
#     model_choices = ["Random Forest", "Support Vector Machine", "Multinomial Naive Bayes", "Adaptive Random Forest",
#                      "Hoeffding Tree", "Perceptron Mask"]
#     model_choice = st.selectbox("Select a model", model_choices)
#
#     if model_choice in ["Adaptive Random Forest", "Hoeffding Tree", "Perceptron Mask"]:
#         drift_detection_algorithm_choices = ["ADWIN", "DDM", "EDDM"]
#
#     # Define the preprocessing function for text data
#     nlp = spacy.load('en_core_web_sm')
#
#     def preprocess_text(text):
#         doc = nlp(text)
#         tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
#         return ' '.join(tokens)
#
#     # Define a function to train the selected classification model
#     def train_model(df, model_choice):
#         # Preprocess the text data
#         with st.spinner('Training model....'):
#             df['cleaned_text'] = df['text_'].apply(preprocess_text)
#
#             # Split the dataset into training and testing sets
#             X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2,
#                                                                 random_state=42)
#
#         with st.spinner('This might take few minutes....'):
#             # Convert text data to numerical features using TfidfVectorizer
#             vectorizer = TfidfVectorizer()
#             X_train_vec = vectorizer.fit_transform(X_train)
#             X_test_vec = vectorizer.transform(X_test)
#
#             # Train a classification model on the training data
#             if model_choice == "Random Forest":
#                 clf = RandomForestClassifier()
#             elif model_choice == "Support Vector Machine":
#                 clf = SVC()
#             elif model_choice == "Multinomial Naive Bayes":
#                 clf = MultinomialNB()
#             elif model_choice == "Adaptive Random Forest":
#                 clf = AdaptiveRandomForestClassifier(drift_detection_algorithm=ADWIN)
#             elif model_choice == "Hoeffding Tree":
#                 clf = HoeffdingTreeClassifier(drift_detection_algorithm=DDM)
#             elif model_choice == "Perceptron Mask":
#                 clf = PerceptronMask(drift_detection_algorithm=EDDM)
#
#             clf.fit(X_train_vec, y_train)
#
#             # Evaluate the performance of the trained model on the testing data
#             score = clf.score(X_test_vec, y_test)
#             st.write('Accuracy:', score)
#
#             # Save the trained model as a binary file
#             model_folder = 'C:/Users/Akshay/Desktop/flipkart reviews/Trained Models'
#             filename = os.path.join(model_folder, model_choice.lower().replace(" ", "_") + '_model.sav')
#             vectorizer_filename = os.path.join(model_folder, model_choice.lower().replace(" ", "_") + '_vectorizer.sav')
#             pickle.dump(clf, open(filename, 'wb'))
#             pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))
#             st.write("Trained model saved as", filename)
#
#     # Show the train button to start the training process
#     # Show the train button to start the training process
#     if uploaded_file is not None:
#         df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\Training\reviews.csv")
#
#         if st.button("Train"):
#             train_model(df, model_choice)


def training():
    st.header("Training")
    folder_path = 'C:/Users/Akshay/Desktop/flipkart reviews/Training'
    files = os.listdir(folder_path)
    uploaded_file = st.selectbox('Select a file', files)
    print(uploaded_file)
    # Create a dropdown menu to select a classification model
    model_choice = st.selectbox("Choose a classification model",
                                ["Random Forest", "Support Vector Machine", "Multinomial Naive Bayes","DDM ADWIN"])

    # Define the preprocessing function for text data
    nlp = spacy.load('en_core_web_sm')

    def preprocess_text(text):
        doc = nlp(text)
        tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
        return ' '.join(tokens)

    # Define a function to train the selected classification model
    def train_model(df, model_choice):
        # Preprocess the text data
        with st.spinner('Training model....'):
            df['cleaned_text'] = df['text_'].apply(preprocess_text)

            # Split the dataset into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2,
                                                                random_state=42)

        with st.spinner('This might take few minutes....'):
            # Convert text data to numerical features using TfidfVectorizer
            vectorizer = TfidfVectorizer()
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)

            # Train a classification model on the training data
            if model_choice == "Random Forest":
                clf = RandomForestClassifier()
            elif model_choice == "Support Vector Machine":
                clf = SVC()
            elif model_choice == "Multinomial Naive Bayes":
                clf = MultinomialNB()
            elif model_choice == "DDM CONCEPT DRIFT MODEL":
            # Initialize DDM and ADWIN drift detectors
                clf = DDM()


                # Train the model incrementally and monitor for concept drift
                for i in range(len(X_train_vec.shape[0])):
                    instance = X_train_vec[i]
                    label = y_train[i]

                    # Update the drift detectors
                    clf.add_element(instance, label)
                    clf.add_element(instance, label)
                    if clf.detected_change():
                        st.write("DDM detected concept drift at instance:", i)
                        # Update the model with the current instance

                    else:
                        clf.fit(X_train_vec, y_train)

            # Evaluate the performance of the trained model on the testing data

            score = clf.score(X_test_vec, y_test)
            st.write('Accuracy:', score)

            # Save the trained model as a binary file
            model_folder = 'C:/Users/Akshay/Desktop/flipkart reviews/Trained Models'
            filename = os.path.join(model_folder, model_choice.lower().replace(" ", "_") + '_model.sav')
            vectorizer_filename = os.path.join(model_folder, model_choice.lower().replace(" ", "_") + '_vectorizer.sav')
            pickle.dump(clf, open(filename, 'wb'))
            pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))
            st.write("Trained model saved as", filename)

    # Show the train button to start the training process
    if uploaded_file is not None:
        df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\Training\reviews.csv")

        if st.button("Train"):
            train_model(df, model_choice)







# def training():
#     st.header("Training")
#     folder_path = 'C:/Users/Akshay/Desktop/flipkart reviews/Training'
#     files = os.listdir(folder_path)
#     uploaded_file = st.selectbox('Select a file', files)
#     print(uploaded_file)
#     # Create a dropdown menu to select a classification model
#     model_choice = st.selectbox("Choose a classification model",
#                                 ["Random Forest", "Support Vector Machine", "Multinomial Naive Bayes"])
#
#     # Define the preprocessing function for text data
#     nlp = spacy.load('en_core_web_sm')
#
#     def preprocess_text(text):
#         doc = nlp(text)
#         tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
#         return ' '.join(tokens)
#
#     # Define a function to train the selected classification model
#     def train_model(df, model_choice):
#         # Preprocess the text data
#         with st.spinner('Training model....'):
#             df['cleaned_text'] = df['text_'].apply(preprocess_text)
#
#             # Split the dataset into training and testing sets
#             X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['label'], test_size=0.2,
#                                                                 random_state=42)
#
#         with st.spinner('This might take few minutes....'):
#             # Convert text data to numerical features using TfidfVectorizer
#             vectorizer = TfidfVectorizer()
#             X_train_vec = vectorizer.fit_transform(X_train)
#             X_test_vec = vectorizer.transform(X_test)
#
#             # Train a classification model on the training data
#             if model_choice == "Random Forest":
#                 clf = RandomForestClassifier()
#             elif model_choice == "Support Vector Machine":
#                 clf = SVC()
#             elif model_choice == "Multinomial Naive Bayes":
#                 clf = MultinomialNB()
#             clf.fit(X_train_vec, y_train)
#
#             # Evaluate the performance of the trained model on the testing data
#             score = clf.score(X_test_vec, y_test)
#             st.write('Accuracy:', score)
#
#             # Save the trained model as a binary file
#             model_folder = 'C:/Users/Akshay/Desktop/flipkart reviews/Trained Models'
#             filename = os.path.join(model_folder, model_choice.lower().replace(" ", "_") + '_model.sav')
#             vectorizer_filename = os.path.join(model_folder, model_choice.lower().replace(" ", "_") + '_vectorizer.sav')
#             pickle.dump(clf, open(filename, 'wb'))
#             pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))
#             st.write("Trained model saved as", filename)
#
#     # Show the train button to start the training process
#     if uploaded_file is not None:
#         df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\Training\reviews.csv")
#
#         if st.button("Train"):
#             train_model(df, model_choice)
