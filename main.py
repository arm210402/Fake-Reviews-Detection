import streamlit as st
import os
from home_page import home_page
from training import training
from testing import test_model

st.sidebar.title("Navigation")
clicked = True

# Set a boolean variable to True
clicked = True

# Add a navbar with buttons
nav = st.sidebar.radio(".",["Scrap Reviews", "Testing", "Training"],label_visibility='collapsed')

# Show content based on the selected button in the navbar
if nav == "Scrap Reviews":
    home_page()
    clicked = False
elif nav == "Testing":
    test_model()
    clicked = False
elif nav == "Training":
    training()
    clicked = False


# html_code = """
# <h1 style="color: pink;">This is a heading</h1>
# <p>This is a paragraph</p>
# """
#
# st.write(html_code, unsafe_allow_html=True)
