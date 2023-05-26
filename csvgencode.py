import streamlit as st
import csv

from url_modifier import url_modifier
import requests
from bs4 import BeautifulSoup


# Flipkart Review scrapping
def get_reviews(start_page: int, end_page: int, url: str) -> list[tuple[str, str]]:
    url_i = url
    url_f = url_modifier(url_i)
    print(url_f)
    reviews = []
    prev_rev = ""
    for page in range(start_page, end_page + 1):
        url = f"{url_f}{page}"
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.content, "html.parser")
            review_containers = soup.select(".t-ZTKy")

            rating_containers = soup.select("._3LWZlK")
            # print("Rating container: ", rating_containers)

            # for i in range(len(review_containers)):
            #     review = review_containers[i].text.strip()
            #
            #     rating = rating_containers[i].text.strip()
            #     reviews.append((review, rating))

            # Skip the first rating container (the average rating)
            for i in range(1, len(review_containers) - 1):
                review = review_containers[i].text.strip()

                rating = rating_containers[i].text.strip()
                if i == 1:
                    prev_rev = review
                else:
                    reviews.append((prev_rev, rating))
                    prev_rev = review
        except Exception as e:
            st.error(f"Invalid URL")
    return reviews

    # Example usage:
    # Product ID for Product on Flipkart
