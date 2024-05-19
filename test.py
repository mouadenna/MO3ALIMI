import streamlit as st
import json
import os
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from dotenv import load_dotenv
load_dotenv() 
import google.generativeai as genai
from utils import *


genai.configure(api_key="AIzaSyBnSJKGXExGs_-0rDUQH3VwHEKoinJOwG8")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
model = genai.GenerativeModel('gemini-pro')
prompt="""
Generate an elementary math problem in JSON format with the following structure:

{
  "Problem": "Your math problem here",
  "choice1": "First choice",
  "choice2": "Second choice",
  "choice3": "Third choice",
  "answer": "Correct answer"
}

For example, the output should look like this:

{
  "Problem": "What is the sum of 8 and 5?",
  "choice1": "10",
  "choice2": "12",
  "choice3": "13",
  "answer": "13"
}
"""
response = generate_answer(prompt)

jsonfile=json.loads(response.text)

# data = {
#   "name": "Alice",
#   "age": 30,
#   "city": "New York",
#   "hobbies": ["reading", "hiking", "coding"]
# }
# data = [  # Wrap in a list for compatibility
#       {
#           "question": "What is the capital of France?",
#           "options": ["London", "Paris", "Berlin"],
#           "answer": "Paris",
#           "information": "Paris is the capital and most populous city of France."
#       }
#   ]

def run():
    st.set_page_config(
        page_title="Streamlit Quiz App",
        page_icon="❓",
    )

    # Custom CSS for the buttons
    st.markdown("""
    <style>
    div.stButton > button:first-child {
        display: block;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session variables if they do not exist
    default_values = {'current_index': 0, 'current_question': 0, 'score': 0, 'selected_option': None, 'answer_submitted': False}
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Load quiz data
    quiz_data_path = jsonfile


    def restart_quiz():
        st.session_state.current_index = 0
        st.session_state.score = 0
        st.session_state.selected_option = None
        st.session_state.answer_submitted = False

    def submit_answer():
        # Check if an option has been selected
        if st.session_state.selected_option is not None:
            # Mark the answer as submitted
            st.session_state.answer_submitted = True
            # Check if the selected option is correct
            if st.session_state.selected_option == quiz_data[st.session_state.current_index]['answer']:
                st.session_state.score += 10
        else:
            # If no option selected, show a message and do not mark as submitted
            st.warning("Please select an option before submitting.")

    def next_question():
        st.session_state.current_index += 1
        st.session_state.selected_option = None
        st.session_state.answer_submitted = False

    # Title and description
    st.title("Streamlit Quiz App")

    # Progress bar
    progress_bar_value = (st.session_state.current_index + 1) / len(quiz_data)
    st.metric(label="Score", value=f"{st.session_state.score} / {len(quiz_data) * 10}")
    st.progress(progress_bar_value)

    # Display the question and answer options
    question_item = quiz_data[st.session_state.current_index]
    st.subheader(f"Question {st.session_state.current_index + 1}")
    st.title(f"{question_item['question']}")
    st.write(question_item['information'])

    st.markdown(""" ___""")

    # Answer selection
    options = question_item['options']
    correct_answer = question_item['answer']

    if st.session_state.answer_submitted:
        for i, option in enumerate(options):
            label = option
            if option == correct_answer:
                st.success(f"{label} (Correct answer)")
            elif option == st.session_state.selected_option:
                st.error(f"{label} (Incorrect answer)")
            else:
                st.write(label)
    else:
        for i, option in enumerate(options):
            if st.button(option, key=i, use_container_width=True):
                st.session_state.selected_option = option

    st.markdown(""" ___""")

    # Submission button and response logic
    if st.session_state.answer_submitted:
        if st.session_state.current_index < len(quiz_data) - 1:
            st.button('Next', on_click=next_question)
        else:
            st.write(f"Quiz completed! Your score is: {st.session_state.score} / {len(quiz_data) * 10}")
            if st.button('Restart', on_click=restart_quiz):
                pass
    else:
        if st.session_state.current_index < len(quiz_data):
            st.button('Submit', on_click=submit_answer)

if __name__ == "__main__":
    run()
