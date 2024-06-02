# MO3ALIMI

![MO3ALIMI Logo](logo.png)

## Introduction
MO3ALIMI is a platform designed to help illiterate adults learn the basics of literacy. The platform focuses on alphabets, writing, reading, and basic numeracy. Users receive personalized quizzes that assist them in learning and practicing simultaneously.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [User Interface](#user-interface)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

## Features
- **Alphabet Learning**: Users can learn and practice alphabets through interactive lessons.
- **Writing Practice**: Guided writing exercises to improve writing skills.
- **Reading Practice**: Reading exercises to enhance reading ability.
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
To start the application, run:
```bash
streamlit run app.py
```
Follow the on-screen instructions to begin using the platform.

## User Interface
The main interface includes:
- **Homepage**: Welcoming screen with a logo and title.
- **Phonics**: Interactive lessons for learning alphabets and phonics.
- **Numeracy**: Lessons and quizzes on basic numeracy.
- **Writing**: Guided exercises to practice writing.
- **Quizzes Master**: Personalized quizzes to test and reinforce learning.

### Sidebar
The sidebar includes:
- Customizable sidebar with a title "MO3ALIMI sidebar".
- Option to select different subjects to start learning.

### Warnings and Alerts
- A warning message prompting users to select a subject to start learning.

## Configuration
Configuration options can be set in the `config.yaml` file. Here, you can configure settings for the APIs, quiz difficulty, and other parameters.

## Dependencies
MO3ALIMI relies on several external libraries and services:
- Python 3.8+
- Streamlit
- Gemini API
- Mistral
- Whisper
- Google TTS
- Stable Diffusion
- Mini LLM

Refer to `requirements.txt` for a full list of dependencies.

## Examples
Here are a few examples of how users can interact with MO3ALIMI:

1. **Alphabet Lesson**:
    - Users are presented with letters and corresponding images.
    - Audio guidance is provided through text-to-speech.

2. **Writing Practice**:
    - Users trace letters on the screen.
    - Immediate feedback is given on accuracy.



## Contributors
- **Mouad Ennasiry** - [mouadenna](https://github.com/mouadenna)
- **Salim el mardi** - [contributorusername](https://github.com/salimelmardi)
- **Reda El kate** - [contributorusername](https://github.com/redaelkate)


## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
