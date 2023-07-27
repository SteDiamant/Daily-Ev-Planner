import streamlit as st
import PIL.Image as Image
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
st.set_page_config(
    page_title="MainPage",
    page_icon="👋",
    layout="wide",
    
)

def load_image(image_file):
    return Image.open(image_file)

def main():
    st.title("Personal Information")
    st.subheader("Name: Stelios")
    st.subheader("Surname: Diamantopoulos")
    st.subheader("Age: 28")
    c1,c2 = st.columns(2)
    
    
    st.markdown("**Motivation**")
    st.markdown("""
                    $1.01^{365}=37.8$ <>Small Consistent Effort\n
                    $0.99^{365}=0.03$ <>Lead to Great Results
                    """)
    
    st.image(load_image(r"imgs\profile.png"), width=1000)
    st.download_button(
        label="Download CV",
        data=open("data/Resume___CV.pdf", "rb").read(),
        file_name="StylianosDiamantopoulosCV.pdf"
    )

    st.markdown("""Hey there! I'm Stelios Diamantopoulos, a 27-year-old electrical and electronic engineering student from Greece, now based in the Netherlands. I'm passionate about data analytics and business, having recently completed a fascinating minor in the field.

Driven by the belief that daily consistent effort leads to extraordinary results, I'm always seeking new challenges and opportunities. When not immersed in my studies, you'll find me indulging in sports, martial arts, chess, and programming.

I've had enriching experiences as an intern at Rosen Group B.V., a teacher assistant at Saxion University, and an electronic engineer at Roboteam Twente.

My projects, like the 'Daily Ev Planner' and 'Generating SVG Defects for MEM Device Visual Inspection,' reflect my dedication to innovative solutions.

Certified in Matlab Basics, Data Analytics, and Advanced Business Intelligence, I'm committed to continuous learning and growth.""")

    st.markdown("""
### Optimize Your EV Charging with Daily EV Planner! ⚡🚗

Are you looking to maximize the efficiency of your electric vehicles (EVs) for a day?
The "Daily EV Planner" is here to help!

This powerful tool allows you to plan and optimize the charging and discharging behavior of your EVs, ensuring you get the most out of their performance. It provides valuable insights into power consumption, battery health, and power exchange analysis.

**Key Features:**
- Streamlined and user-friendly interface
- Daily power exchange analysis
- Battery view for detailed information
- Visualize EV performance over time with interactive plots

**Data Sources:**
- EV data generated from real-world scenarios
- Production data retrieved from Aardenhuizen region in Oolst, Netherlands

Don't miss this opportunity to optimize your EV fleet and reduce energy costs! 🌱💡
""")
    # Contact Footer
    st.title("Contact Me")
    st.subheader("Email: stdiamant95@gmail.com")
    st.subheader("LinkedIn: https://www.linkedin.com/in/stelios-diamantopoulos-17783820b")
    st.subheader("GitHub: https://github.com/SteDiamant")
    st.subheader("Portfolio: https://stediamant.streamlit.app/")
if __name__ == "__main__":
    main()