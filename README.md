# MO3ALIMI

<img src="logo.png" alt="MO3ALIMI Logo" width="200"/>

## Introduction
MO3ALIMI is a platform designed to help illiterate adults learn the basics of literacy. The platform focuses on alphabets, writing, reading, and basic numeracy. Users receive personalized quizzes that assist them in learning and practicing simultaneously.
the project is available as well on [huggingface](https://huggingface.co/spaces/mouadenna/MO3ALIMI)

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [User Interface](#user-interface)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)

## Features
- **Phonics Practice**: Users can learn and practice phonics through interactive lessons.
- **Writing Practice**: Guided writing exercises to improve writing skills. We created an interface to teach users how to write Arabic letters. We trained a Convolutional Neural Network (CNN) for this, but it was not efficient in real-life applications due to challenges with varying angles and non-centered Arabic letters.
- **Basic Numeracy**: Lessons and quizzes on basic numeracy skills.
- **Personalized Quizzes**: Tailored quizzes to reinforce learning and practice.
- **Visual Learning**: Images generated to support visual learning.

## Technologies Used
- **Gemini API**: Used to generate lessons dynamically.
- **Mistral**: Facilitates user interactions.
- **Whisper**: Provides speech-to-text functionality.
- **Google TTS**: Enables text-to-speech capabilities.
- **Stable Diffusion**: Generates images to support visual learning.
- **Mini LLM**: Assists Stable Diffusion in image generation.

<img src="Concept map.png" alt="MO3ALIMI Logo" width="1000"/>


### Experimental Technologies
- **RAG System**: Attempted to use a Retrieval-Augmented Generation system, but it did not yield good results.
- **Fine-Tuning**: Tried fine-tuning models, but we did not find a suitable dataset to finetune on.

## Installation
To install MO3ALIMI, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/mo3alimi.git
    ```
2. Navigate to the project directory:
    ```bash
    cd mo3alimi
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
To start the application, you need a GPU to run the application efficiently. The application has been successfully tested on Hugging Face.
Dont' forget to add you API's key.

To run the application, use:
```bash
streamlit run app.py
```

## User Interface
The main interface includes:
- **Homepage**: Welcoming screen.
- **Phonics**: Interactive lessons for learning alphabets and phonics.
- **Numeracy**: Lessons and quizzes on basic numeracy.
- **Writing**: Guided exercises to practice writing.
- **Quizzes Master**: Personalized quizzes to test and reinforce learning.


## Examples
Here are a few examples of how users can interact with MO3ALIMI:

1. **Alphabet Lesson**:
    - Users are presented with letters and corresponding images.
    - Audio guidance is provided through text-to-speech as well as the user can respond with voice.

2. **Writing Practice**:
    - Users trace letters on the screen.
    - Immediate feedback is given.



## Team members:
- **Mouad Ennasiry** - [Mouad ennasiry](https://github.com/mouadenna)
- **Salim el mardi** - [Salimelmardi](https://github.com/salimelmardi)
- **Reda El kate** - [redaelkate](https://github.com/redaelkate)
