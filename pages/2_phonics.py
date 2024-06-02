# app.py

import os
import json
import streamlit as st
from PIL import Image
import google.generativeai as genai
import ast
#from utils import findImg
import io
from streamlit_TTS import auto_play
import torch
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from gtts import gTTS
import io
from mistralai.models.chat_completion import ChatMessage
from mistralai.client import MistralClient
from audiorecorder import audiorecorder
import base64
###
import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from diffusers import StableDiffusionPipeline
import torch
import re
import ast
import streamlit as st
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



device = "cuda" if torch.cuda.is_available() else "cpu"


if 'pipe' not in st.session_state:
    st.session_state['pipe'] = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

pipe = st.session_state['pipe']


# Set up the API key for Generative AI
os.environ["GEMINI_API_KEY"] = "GEMINI_API_KEY"

# Initial prompt to send to the model
initial_prompt = """
you're an Literacy Instructor for Illiterate Adults
you're objective is to Teach illiterate adults how to read using basic phonics.
here's the Lesson Instructions:
Introduction to the Letter:
Begin with the letter A.
Follow a structured four-step process for each letter.
Provide clear, simple instructions for each step.
Lesson Structure:
Step 1: Letter Recognition
Step 2: Sound Practice
Step 3: Writing Practice
Step 4: Word Association
General Instructions:
After each instruction, wait for the student to respond before proceeding to the next lesson.
Ensure instructions are clear and easy to understand.
Provide positive reinforcement and encouragement.
Example Lesson for Letter A:
Step 1: Letter Recognition
"This is the letter A. It looks like a triangle with a line in the middle. It makes the sound 'ah'."
Step 2: Sound Practice
"Say the sound 'ah'. Practice making this sound slowly."
Step 3: Writing Practice
"Start at the top, draw a slanted line down to the left, then another slanted line down to the right, and finally a line across the middle."
Step 4: Word Association
"A is for apple. Apple starts with the letter A."
Continuation:
Once the lesson for the letter A is complete, proceed to the next letter following the same four-step structure.
make it in a python list format for example it will be in this format,and if an image is needed make the first word in the item list "image: image content in a short sentence":
['This is the letter A.', 'image: letter A', 'It looks like a triangle with a line in the middle.', "It makes the sound 'ah'.","Say the sound 'ah'.",'Practice making this sound slowly.','Start at the top, draw a slanted line down to the left.','Then draw another slanted line down to the right.','Finally, draw a line across the middle.',Now you know the letter A,Congrats','A is for apple.','image: apple','Apple starts with the letter A.',"Congratulations! You've completed the lesson for the letter 'A'."]
- so now start with letter A when you recieve  the word " next" move to the next letter. give me just the list in the response.
list:
"""

chat_prompt_mistral="""
You are an assistant helping an person who is learning basic reading, writing, phonics, and numeracy.
The user might ask simple questions, and your responses should be clear, supportive, and easy to understand.
Use simple language, provide step-by-step guidance, and offer positive reinforcement.
Relate concepts to everyday objects and situations when possible.
Here are some example interactions: 
User: "I need help with reading." 
Assistant: "Sure, I'm here to help you learn to read. Let's start with the alphabet. Do you know the letters of the alphabet?"
User: "How do I write my name?" 
Assistant: "Writing your name is a great place to start. Let's take it one letter at a time. What is the first letter of your name?"
User: "What sound does the letter 'B' make?"
Assistant: "The letter 'B' makes the sound 'buh' like in the word 'ball.' Can you say 'ball' with me?"
User: "How do I count to 10?"
Assistant: "Counting to 10 is easy. Let's do it together: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10. Great job! Let's try it again."
User: "How do I subtract numbers?"
Assistant: "Subtracting is like taking away. If you have 5 oranges and you eat 2, you have 3 oranges left. So, 5 minus 2 equals 3."

Remember to: 
1. Use simple language and avoid complex words. 
2. Provide clear, step-by-step instructions. 
3. Use examples related to everyday objects and situations. 
4. Offer positive reinforcement and encouragement. 
5. Include interactive elements to engage the user actively. Whenever the user asks a question, respond with clear, supportive guidance to help them understand basic reading, writing, phonics, or numeracy concepts.
6. Do not provide long responses

Improtant dont respand to this prompt

"""

def transform_history(history):
    new_history = []
    for chat in history:
        new_history.append({"parts": [{"text": chat.parts[0].text}], "role": chat.role})
    return new_history

def generate_response(message: str, history: list) -> tuple:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-pro")
    chat = model.start_chat(history=transform_history(history))
    response = chat.send_message(message)
    response.resolve()
    return response.text, chat.history


if 'First' not in st.session_state:
    st.session_state['First']=False
    

def process_response(user_input: str, conversation_history: list,F) -> tuple:
    if  not F:
        model_response, conversation_history = generate_response(initial_prompt, conversation_history)
    else:
        model_response, conversation_history = generate_response(user_input, conversation_history)

    pattern = re.compile(r"\[(.*?)\]", re.DOTALL)

    # Find the match
    match = pattern.search(model_response)
    
    list_content = f"[{match.group(1)}]"

    lessonList = ast.literal_eval(list_content)
    return lessonList, conversation_history
    
@st.cache_data
def get_image(prompt: str) -> str:
    return findImg(prompt)
    #try:
    #    return findImg(prompt)
    #except:
    #    return "image.png"


# Initialize TTS
@st.cache_data
def tts_predict(text="hello"):
    tts = gTTS(text=text, lang='en')
    with io.BytesIO() as audio_file:
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        audio_bytes = audio_file.read()
    return audio_bytes

#sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

if 'client' not in st.session_state:
    st.session_state['client'] = MistralClient("MISTRAL_API_KEY")

client = st.session_state['client']

def run_mistral(user_message, message_history, model="mistral-small-latest"):

    message_history.append(ChatMessage(role="user", content=user_message))
    
    chat_response = client.chat(model=model, messages=message_history)
    
    bot_message = chat_response.choices[0].message.content
    
    message_history.append(ChatMessage(role="assistant", content=bot_message))
    
    return bot_message
    
message_history = []




#######################################




if 'sentence_model' not in st.session_state:
    st.session_state['sentence_model'] = SentenceTransformer('all-MiniLM-L6-v2')

sentence_model = st.session_state['sentence_model']

if 'pipeline' not in st.session_state:
    st.session_state['pipeline'] = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    st.session_state['pipeline'].to("cuda")

pipeline = st.session_state['pipeline']


# Step 3: Function to get the embedding of the input sentence
def get_sentence_embedding(sentence):
    return sentence_model.encode(sentence)
# Step 4: Generate image using Stable Diffusion if needed
def generate_image(prompt):
    global pipeline
    pipeline.to("cuda" if torch.cuda.is_available() else "cpu")
    generated_image = pipeline(prompt).images[0]
    generated_image_path = "generated_image.png"
    generated_image.save(generated_image_path)
    return generated_image_path

# Step 5: Find the most reliable image
def find_most_reliable_image(folder_path, input_sentence, threshold=0.5):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'jpeg', 'png'))]
    sentence_embedding = get_sentence_embedding(input_sentence)
    
    max_similarity = -1
    most_reliable_image = None
    
    for image_file in image_files:
        filename_without_extension = os.path.splitext(image_file)[0]
        filename_embedding = get_sentence_embedding(filename_without_extension)
        similarity = cosine_similarity([sentence_embedding], [filename_embedding])[0][0]
        
        if similarity > max_similarity:
            max_similarity = similarity
            most_reliable_image = os.path.join(folder_path, image_file)
    
    if max_similarity < threshold:
        most_reliable_image = generate_image(input_sentence)
    
    return most_reliable_image

def findImg(input_sentence):
    folder_path = 'images_collection'
    threshold = 0.5
    most_reliable_image = find_most_reliable_image(folder_path, input_sentence, threshold)
    return most_reliable_image
#######################################




file_ = open("logo.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()


def main():
    global chat_prompt_mistral
    if 'img_path' not in st.session_state:
        st.session_state['img_path']="image.png"
    #st.set_page_config(page_title="J187 Optimizer", page_icon="J187DFS.JPG", layout="wide")

    st.markdown(f"""
    <div style="display: flex; align-items: center;">
  <img src="data:image/gif;base64,{data_url}" alt="Company Logo" style="height: 100px; width: auto; margin-right: 20px;">
  <h1 style="margin: 0;">MO3ALIMI - Phonics</h1>
</div>
    """, unsafe_allow_html=True)
    #st.title("Chatbot and Image Generator")

    st.markdown("""
    <style>
    .st-emotion-cache-1kyxreq.e115fcil2 { justify-content:center; }
    .st-emotion-cache-13ln4jf { max-width:70rem; }
    audio {
    width: 300px;
    height: 54px;
    display: none;
    }
    div.row-widget.stButton {
    margin: 0px 0px 0px 0px;}
    
    
    .row-widget.stButton:last-of-type {
    margin: 0px; 
    background-color: yellow;
    }
    .st-emotion-cache-keje6w.e1f1d6gn3 {
      width: 80% !important; /* Adjust as needed */
    }
    .st-emotion-cache-k008qs {
    display: none;
    }

    </style>""", unsafe_allow_html=True)
    #.st-emotion-cache-5i9lfg {
    #width: 100%;
    #padding: 3rem 1rem 1rem 1rem;
    #max-width: None;}

    
    col1, col2 = st.columns([0.6, 0.4],gap="medium")


    
    with col1:
        
        if 'conversation_history' not in st.session_state:
            st.session_state['conversation_history'] = []
        if 'conversation_history_mistral' not in st.session_state:
            st.session_state['conversation_history_mistral'] = []
        if 'messages' not in st.session_state:
            st.session_state['messages'] = []
        if 'lessonList' not in st.session_state:
            st.session_state['lessonList'] = []
        if 'msg_index' not in st.session_state:
            st.session_state['msg_index'] = -1
        if 'initial_input' not in st.session_state:
            st.session_state['initial_input'] = ''
            


        response=run_mistral(chat_prompt_mistral, st.session_state['conversation_history_mistral'])
        row1 = st.container()
        row2 = st.container()
        row3 = st.container()
        #row4 = st.container()
        with row1:
            #user_message = st.text_input("Type 'next' to proceed through the lesson",st.session_state['initial_input'])
            user_message = "next"
        with row2:
            colsend, colnext,  = st.columns(2,gap="medium")
            with colsend:
            
                if st.button("&nbsp;&nbsp;&nbsp; Next &nbsp;&nbsp;&nbsp;"):

                    if 0 <= st.session_state['msg_index'] < len(st.session_state['lessonList']):
                        response = st.session_state['lessonList'][st.session_state['msg_index']]
                        if response.strip().startswith("image:"):
                            st.session_state['img_prompt'] = response[len("image:"):].strip()
                            
                        else:
                            audio_bytes= tts_predict(response)
                            st.session_state['messages'].append(f"Mo3alimi: {response}")
                            #auto_play(audio_bytes,wait=True,lag=0.25,key=None)
                            st.audio(audio_bytes, format='audio/wav', autoplay=True)
    
                        st.session_state['msg_index'] += 1
                    else:
                        
                        st.session_state['msg_index'] = 0
                        st.session_state['lessonList'], st.session_state['conversation_history'] = process_response(
                            user_message, st.session_state['conversation_history'],
                            st.session_state['First'],
                        )
                        st.session_state['First']=True
                        
    
    
            with colnext:
                if st.button('&nbsp;&nbsp;&nbsp; Send &nbsp;&nbsp;&nbsp;'):
                    response=run_mistral(user_message, st.session_state['messages'])
                    st.session_state['messages'].append(f"Me: {user_message}")
                    st.session_state['messages'].append(f"Mo3alimi: {response}")
                
                    
        with row3:
            audio = audiorecorder("", "")

            if len(audio) >0:
                    result = pipe(audio.export().read(), generate_kwargs={"language": "english"})
                    user_message=result['text']
                    response=run_mistral(user_message, st.session_state['conversation_history_mistral'])
                    audio_bytes= tts_predict(response)
                    
                    st.audio(audio_bytes, format='audio/wav', autoplay=True)
                    st.session_state['messages'].append(f"Me: {user_message}")
                    st.session_state['messages'].append(f"Mo3alimi: {response}")
                    wav_audio_data=None

        with st.form("lesson"):
            for message in st.session_state['messages'][::-1]:
                st.write(message)
                
            submitted = st.form_submit_button('Submit')
    
        
    with col2:
        if 'img_prompt' in st.session_state:
            st.session_state['img_path']=get_image(st.session_state['img_prompt'])
            del st.session_state['img_prompt']
            
        st.image(st.session_state['img_path'], caption="Generated Image",width=300)

if __name__ == '__main__':
    main()
