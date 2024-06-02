
import json as js
import os
import streamlit as st
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import json
import base64
import torch
from transformers import pipeline
from diffusers import StableDiffusionPipeline
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from diffusers import StableDiffusionPipeline


import io
from gtts import gTTS
import ast
########################################
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
########################################

@st.cache_data
def get_image(prompt: str) -> str:
    return findImg(prompt)
    #try:
    #    return findImg(prompt)
    #except:
    #    return "image.png"



prompt = """
you're an illiteracy tutot for illiterate people these people just learned the basic like they learn letters, your task is to:
-Generate a python lsit contians multiple-choice question with the following format, Create 10 questions:

- "question": The text of the question.
- "incorrect_answers": A list of possible answer choices which are incorrect.
- "correct_answer": The correct answer. :
- "image": description of image if needed, else keep it empty

Objective: Identify the correct letter to complete a word.

Objective: Match words with their correct images.

Objective: Identify the first and last letters of a word.

Objective: Identify whether the given letter is a vowel or a consonant.

Objective: Identify the word formed by blending given sounds.

Objective: Identify if the vowel in a word is short or long.

here's exemples:
[
{
  "question": "Complete the word: H__T (Hint: Something you wear on your head)",
  "incorrect_answers": ["A", "O", "U"],
  "correct_answer": "E",
  "image": ""
},
{
  "question": "Which word matches this picture?",
  "incorrect_answers": ["CAR", "BUS", "TRAIN"],
  "correct_answer": "BIKE",
  "image": "Picture of a Bike"
},
{
  "question": "What is the last letter of the word 'MONKEY'?",
  "incorrect_answers": ["M", "O", "K"],
  "correct_answer": "Y",
  "image": ""
},
{
  "question": "Is the vowel sound in 'HOP' short or long?",
  "incorrect_answers": ["Long"],
  "correct_answer": "Short",
  "image": ""
},
{
  "question": "Complete the word: B__L (Hint: A round object that bounces)",
  "incorrect_answers": ["A", "I", "U"],
  "correct_answer": "O",
  "image": ""
},
{
  "question": "Which word matches this picture?",
  "incorrect_answers": ["APPLE", "BANANA", "ORANGE"],
  "correct_answer": "PEAR",
  "image": "Picture of a Pear"
}
]
-generate a python list of 10 quizzes.
quizzes=

"""


QS=[
    {
        "question": "What is the beginning sound of the word 'apple'?",
        "incorrect_answers": ["m", "p", "r"],
        "correct_answer": "a"
    },
    {
        "question": "Which word has the same ending sound as 'sun'?",
        "incorrect_answers": ["moon", "run", "fun"],
        "correct_answer": "bun"
    },
    {
        "question": "Which word rhymes with 'ship'?",
        "incorrect_answers": ["dip", "rip", "sip"],
        "correct_answer": "tip"
    },
    {
        "question": "What is the long vowel sound in the word 'rain'?",
        "incorrect_answers": ["a", "e", "i"],
        "correct_answer": "ai"
    },
    {
        "question": "Which word has the same beginning sound as 'elephant'?",
        "incorrect_answers": ["igloo", "apple", "umbrella"],
        "correct_answer": "egg"
    },
    {
        "question": "Which word has the same ending sound as 'kite'?",
        "incorrect_answers": ["bike", "site", "dike"],
        "correct_answer": "light"
    },
    {
        "question": "What is the short vowel sound in the word 'dog'?",
        "incorrect_answers": ["a", "e", "i"],
        "correct_answer": "o"
    },
    {
        "question": "Which word has the same beginning sound as 'under'?",
        "incorrect_answers": ["apple", "egg", "umbrella"],
        "correct_answer": "up"
    },
    {
        "question": "Which word has the same ending sound as 'bat'?",
        "incorrect_answers": ["cat", "rat", "hat"],
        "correct_answer": "mat"
    },
    {
        "question": "What is the long vowel sound in the word 'boot'?",
        "incorrect_answers": ["a", "e", "i"],
        "correct_answer": "oo"
    }
]


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
@st.cache_data
def tts_predict(text="hello"):
    tts = gTTS(text=text, lang='en')
    with io.BytesIO() as audio_file:
        tts.write_to_fp(audio_file)
        audio_file.seek(0)
        audio_bytes = audio_file.read()
    return audio_bytes
@st.cache_data
def get_mistral_response_injson():
    global prompt
    api_key = "mRLAIaEj5reOROa19MBF3dj21DaSEELl"
    model = "mistral-small-latest"
    client = MistralClient(api_key=api_key)

    messages = [
        ChatMessage(role="user", content=prompt)
    ]

    chat_response = client.chat(
        model=model,
        response_format={"type": "json_object"},
        messages=messages,
    )

    output = chat_response.choices[0].message.content

    return output




@st.cache_data(ttl= 75, max_entries=1)
def get_questions():
    questions = ast.literal_eval(get_mistral_response_injson())
    return questions

def initialize_session_state():
    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
        # st.snow()
    if 'player_score' not in st.session_state:
        st.session_state.player_score = 0

def update_score(player_choice, correct_answer):
    if str(player_choice) == str(correct_answer):
        st.success("It was a correct answer! Great Job! 😁✌")
        st.session_state.player_score += 1
        #st.balloons()
    else:
        st.error("It was an incorrect answer! 😕")

if "page" not in st.session_state:
    st.session_state.page = 0
if "submit_key" in st.session_state and st.session_state.submit_key == True:
    st.session_state.running = True
else:
    st.session_state.running = False

if "running" not in st.session_state:
    st.session_state.running = False

def nextpage(): st.session_state.page += 1
def restart(): st.session_state.page = 0

# set_bg_hack_url()
st.markdown("""<style>description {color: Green;}

.st-emotion-cache-1y4p8pa {
    /* padding: none; */
}

            </style>""",unsafe_allow_html = True)

file_ = open("logo.png", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")

st.markdown(f"""
    <div style="display: flex; align-items: center;">
  <img src="data:image/gif;base64,{data_url}" alt="Company Logo" style="height: 100px; width: auto; margin-right: 20px;">
  <h1 style="margin: 0;">MO3ALIMI</h1>
</div>
    """, unsafe_allow_html=True)

st.markdown("""
<style>
audio {
width: 300px;
height: 54px;
display: none;
}
</style>""", unsafe_allow_html=True)



initialize_session_state()

def calculate_score(player_choice):
    correct_answer = quiz_questions[st.session_state.current_question]["answer"]
    # st.write("inside calculate_score" + str(correct_answer))
    update_score(player_choice, correct_answer)
    st.session_state.current_question += 1


#if 'category'not in st.session_state:
#    st.session_state.category =None
#if st.sidebar.button('&nbsp;'*30+'Phonics'+'&nbsp;'*30, ):
#    st.session_state.category = 'phonics'
#if st.sidebar.button('&nbsp;'*27+'Numeracy'+'&nbsp;'*28, ):
#    st.session_state.category = 'numeracy'
#if st.sidebar.button('&nbsp;'*31+'Writing'+'&nbsp;'*31, ):
#    st.session_state.category = 'Writing'
#if st.sidebar.button('&nbsp;'*30+'Reading'+'&nbsp;'*30, ):
#    st.session_state.category = 'Reading'
#category = st.session_state.category

#category = st.sidebar.selectbox("Category: ", ['Phonics','Numeracy','Writing','Reading'], index= None, placeholder= "Select one: ", disabled=(st.session_state.running))

if 'img_path2' not in st.session_state:
    st.session_state['img_path2']="image.png"
# st.session_state.disable_opt = True
# category = st.sidebar.selectbox("Category: ", list(categories_option.keys()), index= None, placeholder= "Select one: ", disabled=(st.session_state.running))
col1, col2 = st.columns([0.6, 0.4],gap="medium")
with col1:
    QuestionList = get_questions()
    # st.write(QuestionList)
    len_response = len(QuestionList)
    if len_response == 0:
        st.error("Got no question! 😕😔")
        st.stop()
    quiz_questions = []
    for item in range(len_response):
        temp_dict = dict()
        temp_dict['text'] = QuestionList[item].get("question")
        temp_dict['options'] = tuple(QuestionList[item].get("incorrect_answers") + [QuestionList[item].get("correct_answer")])
        temp_dict['answer'] = QuestionList[item].get("correct_answer")
        temp_dict['image'] = QuestionList[item].get("image")

        quiz_questions.append(temp_dict)
    placeholder = st.empty()
    ind = st.session_state.current_question
    if ind > len(quiz_questions):
        st.stop()
    else:



        current_question = quiz_questions[ind]
        st.subheader(quiz_questions[ind]["text"])

        audio_bytes = tts_predict(quiz_questions[ind]["text"])
        st.audio(audio_bytes, format='audio/wav', autoplay=True)

        if quiz_questions[ind]["image"]:
            st.session_state['img_prompt2']=quiz_questions[ind]["image"]


        def play_audio(choice):
            audio_bytes = tts_predict(choice)
            st.audio(audio_bytes, format='audio/wav', autoplay=True)




        player_choice = st.radio("Select your answer:",
                                 options=current_question["options"],
                                 key=f"question_{ind}",
                                 index=None,
                                 disabled=(st.session_state.running))

        try:
            play_audio(player_choice)
        except:
            pass

        submitted =  st.button("Submit", key="submit_key", disabled=(st.session_state.running))
        if submitted:
            calculate_score(player_choice)
            st.markdown("Correct Answer: "+ current_question["answer"])
        # st.empty()
            if st.button("Next",on_click=nextpage,disabled=(st.session_state.page >= 9)):
                if st.session_state.current_question < len(quiz_questions):
                    st.rerun()
            if st.session_state.current_question >= len(quiz_questions):
            # st.session_state.clear
                st.empty()
                st.success("Quiz Finished!")
                st.subheader(f"_Your_ _Score_: :blue[{st.session_state.player_score}] :sunglasses:")
                #st.snow()

with col2:
    if 'img_prompt2' in st.session_state:
        st.session_state['img_path2'] = get_image(st.session_state['img_prompt2'])
        del st.session_state['img_prompt2']

    st.image(st.session_state['img_path2'], caption="Generated Image", width=300)
st.markdown("---")