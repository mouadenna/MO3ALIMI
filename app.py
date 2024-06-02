import streamlit as st
import base64
#st.set_page_config(page_title="Homepage")
st.set_page_config(page_title="Homepage", page_icon="J187DFS.JPG", layout="wide")

file_ = open("logo.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
css=f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/gif;base64,{data_url}" alt="Company Logo" style="height: 100px; width: auto; margin-right: 20px;">
        <h1 style="margin: 0;">
            <span style="color: orange;">Welcome to</span> 
            <span style="color: violet;">MO3ALIMI!</span>
        </h1>
    </div>

"""
st.markdown(css, unsafe_allow_html=True)
def add_logo():
    st.markdown(
        """
        <style>
            [data-testid="stSidebarNav"] {
                /*background-image: url(http://placekitten.com/200/200);*/
                background-repeat: no-repeat;
                #padding-top: 120px;
                background-position: 20px 20px;
            }
            [data-testid="stSidebarNav"]::before {
                content: "MO3ALIMI sidebar";
                margin-left: 20px;
                margin-top: 20px;
                font-size: 29px;
                position: relative;
                top: 0px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
add_logo()


#st.sidebar.title("MO3ALIMI sidebar")
#st.sidebar.markdown("---")

st.warning('Please select a subject to start learning', icon="⚠️")
st.markdown("---")