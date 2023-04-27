import streamlit as st
from PIL import Image

image = Image.open('imgs/image.png')

st.image(image, caption='Aardenhuizen', use_column_width=True)