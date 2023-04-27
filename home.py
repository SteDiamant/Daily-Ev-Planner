import streamlit as st
from PIL import Image

image = Image.open(r'imgs\image.png')

st.image(image, caption='Aardenhuizen', use_column_width=True)