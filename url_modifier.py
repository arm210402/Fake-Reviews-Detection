import streamlit as st
def url_modifier(url):
    try:
        url_modified = url.replace("/p/", "/product-reviews/")
        third_ampersand = url_modified.find('&', url_modified.find('&', url_modified.find('&') + 1) + 1)
        if third_ampersand != -1:
            url_modified = url_modified[:third_ampersand]

        url_modified = url_modified + "&page="
        return url_modified
    except Exception as e:
        st.error(f"Invalid URL")
