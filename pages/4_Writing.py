import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import random
import os


arabic_chars = ['alef','beh','teh','theh','jeem','hah','khah','dal','thal','reh','zain','seen','sheen',
                'sad','dad','tah','zah','ain','ghain','feh','qaf','kaf','lam','meem','noon','heh','waw','yeh']




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



def predict_image(image_path, model_path):
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (32, 32))
        img = img.reshape(1, 32, 32, 1)
        img = img.astype('float32') / 255.0

        pred = model.predict(img)
        predicted_label = arabic_chars[np.argmax(pred)]
        return predicted_label
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None


def get_next_image(folder_path, current_char):
    try:
        current_index = arabic_chars.index(current_char)
        next_index = (current_index + 1) % len(arabic_chars)
        next_char = arabic_chars[next_index]
        image_path = os.path.join(folder_path, f"{next_char}.png")
        return image_path, next_char
    except Exception as e:
        st.error(f"Error loading next image: {e}")
        return None, None


# Load and display a random image
folder_path = "arabic letters"
if 'image_path' not in st.session_state:
    st.session_state.image_path, st.session_state.correct_char = os.path.join(folder_path, f"{arabic_chars[0]}.png"),arabic_chars[0]
col1,col2,col3=st.columns([1,1,1])
with col1:
    if st.session_state.image_path and st.session_state.correct_char:
        st.image(st.session_state.image_path, caption=f"Draw this character: {st.session_state.correct_char}",width=350,)
    else:
        st.error("Error loading the random image.")

if 'counter' not in st.session_state:
    st.session_state.counter = 0

with col2:
    canvas_result = st_canvas(
        fill_color="rgba(255, 255, 255, 0.3)",  # Filled color (white)
        stroke_width=19,  # Stroke width
        stroke_color="#FFFFFF",  # Stroke color (white)
        background_color="#000000",  # Canvas background color (black)
        update_streamlit=True,
        height=400,
        width=400,
        drawing_mode="freedraw",
        key="canvas",
        )
with col3:
    if st.button("Check"):
        if canvas_result.image_data is not None:
            image = canvas_result.image_data
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (32, 32))

            # Save the image temporarily
            temp_image_path = "temp_image.png"
            cv2.imwrite(temp_image_path, image)

            # Load and predict using the model
            model_path = "saved_model.h5"  # Replace with the path to your trained model
            if os.path.exists(model_path):
                predicted_label = predict_image(temp_image_path, model_path)
                if predicted_label:
                    #st.write(f"Predicted Character: {predicted_label}")
                    if predicted_label == st.session_state.correct_char:
                        st.success("You are correct!")
                        st.session_state.image_path, st.session_state.correct_char = get_next_image(folder_path,st.session_state.correct_char)
                        #canvas_result.clear_background()
                    else:
                        st.error("The prediction does not match the displayed character. Try again.")
            else:
                st.error("Model file not found. Please check the model path.")
        else:
            st.write("Please draw something on the canvas.")
    if st.button("Next Character"):
        # Load the next image and clear the canvas
        st.session_state.image_path, st.session_state.correct_char = get_next_image(folder_path,
                                                                                    st.session_state.correct_char)