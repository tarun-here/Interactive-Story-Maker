import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.prompts import PromptTemplate
import nltk
from nltk.stem import WordNetLemmatizer
import streamlit as st
import os

PERSIST_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db") 

nltk.download('punkt')
nltk.download('wordnet')

genai.configure(api_key="GEMINI_API_KEY")
model = genai.GenerativeModel("gemini-1.5-flash")

if "story_worlds" not in st.session_state:
    st.session_state.story_worlds = {}
if "story_id" not in st.session_state:
    st.session_state.story_id = None
if "current_scene" not in st.session_state:
    st.session_state.current_scene = None
if "user_choices" not in st.session_state:
    st.session_state.user_choices = []

story_worlds = st.session_state.story_worlds


def create_story_world(name, setting, time_period, atmosphere):
    story_world_id = len(story_worlds) + 1
    story_worlds[story_world_id] = { 
        "name": name,
        "setting": setting,
        "time_period": time_period,
        "atmosphere": atmosphere,
        "characters": {}
    }
    return story_world_id


def create_character(story_world_id, name, traits, backstory, role):
    print(f"Story world ID in create_character: {story_world_id}")
    character_id = len(story_worlds[story_world_id]["characters"]) + 1
    story_worlds[story_world_id]["characters"][character_id] = {
        "name": name,
        "traits": traits,
        "backstory": backstory,
        "role": role
    }
    return character_id


def update_story_state(story_id, new_scene):
    story_worlds[story_id]["current_scene"] = new_scene


def create_knowledge_graph(story_world_id):
    story_world_data = story_worlds[story_world_id]
    character_data = list(story_world_data["characters"].values())

    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    
    story_world_embedding = embeddings.embed_query(story_world_data["name"]) 
    character_embeddings = [embeddings.embed_query(character["name"]) for character in character_data]

    text_embeddings = [(character["name"], embedding) for character, embedding in zip(character_data, character_embeddings)]

    if text_embeddings:
        vector_store = Chroma.from_texts(
            texts=[text for text, _ in text_embeddings],
            embedding=embeddings,
            ids=[str(i) for i in range(len(text_embeddings))],
            persist_directory=PERSIST_DIRECTORY 
        )
    else:
        vector_store = Chroma.from_texts(texts=[], embedding=embeddings,persist_directory=PERSIST_DIRECTORY)

    vector_store.add_texts(
    texts=[story_world_data["name"]], 
    embedding=embeddings, 
    ids=["story_world"]
    )

    return vector_store


def generate_question(story_world_id, current_scene, user_choices):
    vector_store = create_knowledge_graph(story_world_id)

    prompt_template = """Generate a question based on the following information:

    Current scene: {current_scene}
    User choices: {user_choices}

    Consider the story world's setting, characters, and their relationships.
    """
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["current_scene", "user_choices"])

    combined_input = f"Current scene: {current_scene}\nUser choices: {user_choices}"

    response = model.generate_content(combined_input)
    question = response.text
    return question


def process_user_input(user_input):
    tokens = nltk.word_tokenize(user_input)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas


def generate_response(story_id, user_input):
    story_data = story_worlds[story_id]
    character_data = story_data["characters"]

    prompt_template = """Generate a response based on the following information:

    Story: {story_data}
    User input: {user_input}
    Characters: {character_data}

    Consider the story's setting, characters, and their relationships.
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["story_data", "user_input", "character_data"])

    response = model.generate_content(
        prompt.format(story_data=story_data,
                      user_input=user_input,
                      character_data=character_data))  
    response_text = response.text
    return response_text


# --- Story Creation ---
st.header("Create Your Story World")
with st.form("story_world_form"):
    name = st.text_input("Story World Name:")
    setting = st.text_input("Setting:")
    time_period = st.text_input("Time Period:")
    atmosphere = st.text_input("Atmosphere:")
    if st.form_submit_button("Create Story World"):
        story_world_id = create_story_world(name, setting, time_period,
                                            atmosphere)
        st.session_state.story_id = story_world_id
        st.success(f"Story world created with ID: {story_world_id}")

# --- Character Creation ---
if st.session_state.story_id:
    st.header("Create Your Characters")
    print(f"Story worlds: {story_worlds}")
    with st.form("character_form"):
        name = st.text_input("Character Name:")
        traits = st.text_area("Traits:")
        backstory = st.text_area("Backstory:")
        role = st.text_input("Role:")
        if st.form_submit_button("Create Character"):
            character_id = create_character(st.session_state.story_id, name,
                                            traits, backstory, role)
            st.success(f"Character created with ID: {character_id}")

# --- Interactive Story ---
if st.session_state.story_id:
    st.header("Interactive Story")
    if st.session_state.current_scene is None:
        # Initialize the story
        initial_prompt = "You wake up in a dark room with no memory of how you got there."
        st.write(initial_prompt)
        st.session_state.current_scene = initial_prompt
        update_story_state(st.session_state.story_id,
                          st.session_state.current_scene)
        question = generate_question(st.session_state.story_id,
                                    st.session_state.current_scene,
                                    st.session_state.user_choices)
        st.write(question)
    else:
        user_input = st.text_input("Your Choice:")
        if user_input:
            processed_input = process_user_input(user_input)
            response = generate_response(st.session_state.story_id,
                                        user_input)
            st.write(response)
            st.session_state.current_scene = f"{st.session_state.current_scene} \n{response}"
            update_story_state(st.session_state.story_id,
                              st.session_state.current_scene)
            st.session_state.user_choices.append(user_input) # Add this line to store user choices
            question = generate_question(st.session_state.story_id, st.session_state.current_scene, st.session_state.user_choices)
            st.write(question)