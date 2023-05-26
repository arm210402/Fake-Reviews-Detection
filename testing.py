import os
import pandas as pd
import pickle
import spacy
import streamlit as st
import altair as alt
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import *
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB


# Define the preprocessing function for text data
from clean_data import clean_text


nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct and token.text != 'READ MORE' and token.text != 'read']
    return ' '.join(tokens)

def test_model():
    # Preprocess the text data
    st.header("Testing")

    # Load the saved model
    model_file = r'Trained Models/support_vector_machine_model.sav'
    vectorizer_file = r'Trained Models\multinomial_naive_bayes_vectorizer.sav'
    loaded_model = pickle.load(open(model_file, 'rb'))
    vectorizer = pickle.load(open(vectorizer_file, 'rb'))

    # Load the unlabelled data
    df_unlabeled = pd.read_csv(r'C:\Users\Akshay\Desktop\flipkart reviews\Generated reviews\flipkart_reviews.csv')

    # Preprocess the unlabelled data
    df_unlabeled['cleaned_text'] = df_unlabeled['Review'].apply(clean_text)

    # Convert the unlabelled data to numerical features using the loaded vectorizer
    X_unlabeled_vec = vectorizer.transform(df_unlabeled['cleaned_text'])

    # Use the loaded model to predict the sentiment labels for the unlabelled data
    predicted_labels = loaded_model.predict(X_unlabeled_vec)

    # Add the predicted labels to the unlabelled data and save as a new CSV file
    df_unlabeled['predicted_label'] = predicted_labels
    df_unlabeled.to_csv('predicted_data.csv', index=False)
    df = pd.read_csv(r"C:\Users\Akshay\Desktop\flipkart reviews\predicted_data.csv")

    df_predicted = pd.read_csv("predicted_data.csv")  # Update with the actual path to your predicted CSV file

    # Calculate the average review
    average_rating = df_predicted['Rating'].mean()
    average_rating_formatted = round(average_rating, 1)
    # Display the average review using st.write
    st.markdown(f"<p>Average Rating: <span style='color:#66FF99;'>{average_rating_formatted}</span></p>", unsafe_allow_html=True)


    # Create a bar chart of predicted labels
    label_counts = df["predicted_label"].value_counts()
    chart = alt.Chart(label_counts.reset_index()).mark_bar().encode(
        x=alt.X('index', axis=alt.Axis(title='Predicted Label')),
        y=alt.Y('predicted_label', axis=alt.Axis(title='Count')),
        tooltip=['index', 'predicted_label']
    ).properties(
        width=600,
        height=400
    ).interactive()
    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Display the chart using Streamlit
    st.title("Distribution of Predicted Labels")
    st.altair_chart(chart)

    st.write(df.head(10))

    text = " ".join(review for review in df.cleaned_text)

    # Generate a word cloud image
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

    # Display the image in Streamlit
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    st.pyplot()

    # Select the last two columns and the first 10 rows
    df_new = df[['cleaned_text', 'predicted_label']].head(10)

    # Create a bar chart to show the predicted labels
    fig = px.bar(df_new, x='predicted_label', color='predicted_label',
                 labels={'predicted_label': 'Predicted Label', 'count': 'Count'},
                 title='Predicted Label Distribution for First 10 Reviews')

    # Create a scatter plot to show the cleaned_text
    df_new = df[['cleaned_text', 'predicted_label']].head(10)

    # Create a bar chart to show the predicted labels
    fig = px.bar(df_new, x='predicted_label', color='predicted_label',
                 labels={'predicted_label': 'Predicted Label', 'count': 'Count'},
                 title='Predicted Label Distribution for First 10 Reviews')





