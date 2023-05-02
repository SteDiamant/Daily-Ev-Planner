import streamlit as st
from PIL import Image

charging_image = Image.open('imgs/frontCover.png')
aardenhuizen_image=Image.open('imgs/image.png')
st.markdown('## _Welcome to the Tool to Investigate the impact of 4 Evs at Aardenhuizen region_')
c1,c2 = st.columns([2,1])
with c1:
    st.image(aardenhuizen_image, use_column_width=True)
with c2:
    st.markdown("""The Aardenhuizen community is a small region consisting of 24 households .</br>
                    The community has installed 327 solar panels to generate electricity and 4 electric vehicles (EVs) to promote sustainable transportation.</br>
                    The community members are committed to reducing their carbon footprint by adopting environmentally friendly practices. </br>
                    They prioritize the use of renewable energy sources, such as solar power, to minimize their reliance on non-renewable energy sources.</br>""",unsafe_allow_html=True)
st.markdown('## _Check the DailyEVPlanner on the left sidebar_')