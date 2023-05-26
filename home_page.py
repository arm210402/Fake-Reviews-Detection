import streamlit as st
from csvgencode import get_reviews
import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd
import time
import plotly.express as px


def home_page():
    st.header("Flipkart Product Reviews")
    user_link = st.text_input("Enter Product Url", "")

    # Add a submit button
    if st.button("Submit"):
        try:
            title = requests.get(user_link)
            soup = BeautifulSoup(title.content, "html.parser")
            # product_name = soup.find("h1", class_="product-name").text
            # reviews_count = soup.find("span", class_="_2_R_DZ").text.strip().split("&")[1].strip()

            # spinner
            with st.spinner('Scrapping reviews....'):
                # Replace this with your actual code that takes time to run
                reviews_count = soup.find("span", class_="_2_R_DZ").text.strip().split("&")[1].strip().split()[0].replace(
                    ',',
                    '')
                product_name = soup.find('span', {'class': 'B_NuCI'}).text
                st.write(product_name)
                st.write("Reviews: ", reviews_count)
                # print("Product Name:", product_name)
                print("Number of Reviews:", reviews_count)
                # Replace '20' with any number you want
                reviews = get_reviews(1, 20, user_link)

                with open("Generated reviews/flipkart_reviews.csv", mode="w", encoding="utf-8", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(["Rating", "Review"])
                    for review, rating in reviews:
                        writer.writerow([rating, review])
                    st.write("<p style='color:#66FF99;'>Scrapping finished successfully</p>", unsafe_allow_html=True)

        except Exception as e:
            st.error("Enter a valid URL")


            # Save reviews to a CSV file

            # positive negative average reviews
            # df = pd.read_csv("C:/Users/Akshay/Desktop/flipkart reviews/Generated reviews/flipkart_reviews.csv")
            # Load data
