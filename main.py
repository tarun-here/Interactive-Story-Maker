import os
import time
import uuid
import json
from typing import List, Dict, Any

import streamlit as st
import google.generativeai as genai
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
PERSIST_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")
os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

# Set model name for Gemini
GEMINI_MODEL_NAME = "models/gemini-2.5-flash"

# Embedding model name (sentence-transformers)
EMBEDDING_MODEL_NAME = "all-mpnet-base-v2"

# Number of retrieved documents to inject into prompts
RETRIEVE_K = 5

# ---------------------------
# Initialize/utility functions
# ---------------------------

def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the GEMINI_API_KEY environment variable before running the app.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


def init_embeddings():
    if "embeddings" not in st.session_state:
        st.session_state["embeddings"] = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return st.session_state["embeddings"]


def get_vector_store_for_story(story_id: int):
    """Return a Chroma vector store instance for the given story.
    We cache the loaded vector store in session_state to avoid repeatedly reloading from disk.
    """
    key = f"vector_store_{story_id}"
    embeddings = init_embeddings()
    if key not in st.session_state:
        try:
            vs = Chroma.from_texts(texts=[], embedding=embeddings, persist_directory=PERSIST_DIRECTORY)
        except Exception:
            vs = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        st.session_state[key] = vs
    return st.session_state[key]


# ---------------------------
# Domain data helpers
# ---------------------------

def create_story_world(name: str, setting: str, time_period: str, atmosphere: str) -> int:
    if "story_worlds" not in st.session_state:
        st.session_state.story_worlds = {}
    story_worlds = st.session_state.story_worlds
    story_world_id = len(story_worlds) + 1

    story_worlds[story_world_id] = {
        "id": story_world_id,
        "name": name,
        "setting": setting,
        "time_period": time_period,
        "atmosphere": atmosphere,
        "characters": {},  # id -> character dict
        "scenes": [],      # list of scene strings (chronological)
        "user_inputs": []  # history of raw user inputs
    }

    # Immediately index the new world (world-level document)
    index_story_world_to_chroma(story_world_id)
    return story_world_id


def create_character(story_world_id: int, name: str, traits: str, backstory: str, role: str) -> int:
    story_worlds = st.session_state.story_worlds
    characters = story_worlds[story_world_id]["characters"]
    character_id = len(characters) + 1
    characters[character_id] = {
        "id": character_id,
        "name": name,
        "traits": traits,
        "backstory": backstory,
        "role": role
    }
    # Index the new character into Chroma for retrieval
    index_character_to_chroma(story_world_id, character_id)
    return character_id


# ---------------------------
# Indexing functions (store docs in Chroma)
# ---------------------------

def index_story_world_to_chroma(story_world_id: int):
    """Create or update a single document that captures the story world's metadata.
    We'll store the combined textual representation as a single document.
    """
    story_world = st.session_state.story_worlds[story_world_id]
    embeddings = init_embeddings()
    vector_store = get_vector_store_for_story(story_world_id)

    text = (
        f"Story World: {story_world['name']}\n"
        f"Setting: {story_world['setting']}\n"
        f"Time Period: {story_world['time_period']}\n"
        f"Atmosphere: {story_world['atmosphere']}"
    )

    try:
        vector_store.add_texts(texts=[text], ids=[f"world_{story_world_id}"])
    except Exception:
        pass


def index_character_to_chroma(story_world_id: int, character_id: int):
    story_world = st.session_state.story_worlds[story_world_id]
    char = story_world["characters"][character_id]
    vector_store = get_vector_store_for_story(story_world_id)

    text = (
        f"Character: {char['name']}\n"
        f"Traits: {char['traits']}\n"
        f"Backstory: {char['backstory']}\n"
        f"Role: {char['role']}"
    )

    doc_id = f"char_{character_id}"
    try:
        vector_store.add_texts(texts=[text], ids=[doc_id])
    except Exception:
        pass


def index_scene_to_chroma(story_world_id: int, scene_text: str) -> str:
    """Append scene to story state and index it. Returns the scene id.
    Scenes are small chunks (e.g., paragraph-level) so we can retrieve them later.
    """
    story_world = st.session_state.story_worlds[story_world_id]
    scene_id = f"scene_{len(story_world['scenes']) + 1}"
    story_world["scenes"].append(scene_text)
    vector_store = get_vector_store_for_story(story_world_id)
    try:
        vector_store.add_texts(texts=[scene_text], ids=[scene_id])
    except Exception:
        pass
    return scene_id


def index_user_input_to_chroma(story_world_id: int, user_input: str) -> str:
    story_world = st.session_state.story_worlds[story_world_id]
    story_world["user_inputs"].append(user_input)
    doc_id = f"input_{len(story_world['user_inputs'])}"
    vector_store = get_vector_store_for_story(story_world_id)
    metadata = {"type": "user_input", "timestamp": time.time()}
    try:
        vector_store.add_texts(texts=[user_input], ids=[doc_id])
    except Exception:
        pass
    return doc_id


# ---------------------------
# Retrieval
# ---------------------------

def retrieve_relevant_docs(story_world_id: int, query: str, k: int = RETRIEVE_K) -> List[Dict[str, Any]]:
    """Return top-k relevant documents from Chroma for the query.
    The returned items will be a list of dicts with 'text' and optionally 'id' or 'metadata'.
    """
    vs = get_vector_store_for_story(story_world_id)
    try:
        results = vs.similarity_search(query, k=k)
        normalized = []
        for r in results:
            if isinstance(r, str):
                normalized.append({"text": r})
            elif isinstance(r, dict):
                normalized.append(r)
            else:
                # if it's a Document object or tuple, try to extract text
                try:
                    normalized.append({"text": getattr(r, 'page_content', str(r))})
                except Exception:
                    normalized.append({"text": str(r)})
        return normalized
    except Exception:
        # If similarity_search isn't available, return an empty list to keep the app running
        return []


# ---------------------------
# Prompting utilities
# ---------------------------

def prepare_context_from_retrieval(retrieved: List[Dict[str, Any]]) -> str:
    """Convert retrieved docs into a single context string to inject into the prompt."""
    if not retrieved:
        return ""
    parts = []
    for i, doc in enumerate(retrieved, start=1):
        text = doc.get("text") if isinstance(doc, dict) else str(doc)
        parts.append(f"[Memory {i}]: {text}")
    return "\n".join(parts)


# ---------------------------
# Generation functions
# ---------------------------

def generate_response(story_world_id: int, user_input: str, model) -> str:
    """Generate story continuation using retrieved memories + story state + characters.
    Steps:
      1. Index the user input (so it becomes searchable later)
      2. Retrieve relevant memories from Chroma
      3. Build prompt including: retrieved memories, recent scenes, user input, and characters
      4. Ask Gemini for the next story continuation
      5. Index the new scene into Chroma
    """
    # 1) Index user input
    try:
        index_user_input_to_chroma(story_world_id, user_input)
    except Exception:
        pass

    # 2) Retrieve relevant docs
    retrieved = retrieve_relevant_docs(story_world_id, user_input, k=RETRIEVE_K)
    retrieved_context = prepare_context_from_retrieval(retrieved)

    # 3) Build prompt
    story_world = st.session_state.story_worlds[story_world_id]
    recent_scenes = "\n\n".join(story_world["scenes"][-3:]) if story_world["scenes"] else ""
    characters_summary = []
    for c in story_world["characters"].values():
        characters_summary.append(f"{c['name']} (role: {c['role']}) - {c['traits']}")
    characters_text = "\n".join(characters_summary)

    prompt_parts = [
        "You are a creative story-telling assistant. Produce a vivid continuation of the scene.",
        "Context (retrieved memories):", retrieved_context,
        "Recent Scenes:", recent_scenes,
        "Characters:", characters_text,
        "User Action:", user_input,
        "Produce a short, evocative continuation (1-3 paragraphs) that respects characters and the world."
    ]
    prompt_text = "\n\n".join([p for p in prompt_parts if p])

    # 4) Call Gemini
    try:
        resp = model.generate_content(prompt_text)
        text = resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        text = f"[Error generating response: {e}]"

    # 5) Index the generated scene
    try:
        index_scene_to_chroma(story_world_id, text)
    except Exception:
        pass

    # Append to in-memory scene list
    st.session_state.story_worlds[story_world_id]["scenes"].append(text)
    return text


def generate_question(story_world_id: int, model, user_choices: List[str] = None) -> str:
    """Generate a question to prompt the user for the next choice.
    Uses retrieved memories and the latest story context.
    """
    story_world = st.session_state.story_worlds[story_world_id]
    current_story_text = "\n\n".join(story_world["scenes"]) if story_world["scenes"] else ""
    user_choices = user_choices or story_world.get("user_inputs", [])

    # retrieve based on the latest story text
    seed_query = user_choices[-1] if user_choices else (current_story_text[-512:] if current_story_text else "")
    retrieved = retrieve_relevant_docs(story_world_id, seed_query, k=RETRIEVE_K)
    retrieved_context = prepare_context_from_retrieval(retrieved)

    prompt_parts = [
        "You are a helpful assistant that suggests engaging questions to a player in an interactive fiction.",
        "Context (retrieved memories):", retrieved_context,
        "Current story:", current_story_text,
        "Past user choices:", json.dumps(user_choices[-10:], ensure_ascii=False),
        "Generate 2 short, clear, and distinct next-choice questions the player could take."
    ]
    prompt_text = "\n\n".join([p for p in prompt_parts if p])

    try:
        resp = model.generate_content(prompt_text)
        text = resp.text if hasattr(resp, "text") else str(resp)
    except Exception as e:
        text = f"[Error generating question: {e}]"

    return text


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="RAG Story World", layout="wide")
st.title("RAG-powered Interactive Story World")

# Initialize Gemini + embeddings (fail early)
try:
    model = init_gemini()
except Exception as e:
    st.error(f"Gemini initialization failed: {e}")
    st.stop()

_ = init_embeddings()

# Initialize session state containers if missing
if "story_worlds" not in st.session_state:
    st.session_state.story_worlds = {}
if "story_id" not in st.session_state:
    st.session_state.story_id = None

# --- Story creation form ---
with st.form("story_world_form"):
    st.header("Create or select a Story World")
    name = st.text_input("Story World Name")
    setting = st.text_input("Setting")
    time_period = st.text_input("Time Period")
    atmosphere = st.text_input("Atmosphere / Mood")
    submitted = st.form_submit_button("Create Story World")
    if submitted:
        if not name:
            st.warning("Please provide a story world name.")
        else:
            sid = create_story_world(name, setting, time_period, atmosphere)
            st.session_state.story_id = sid
            st.success(f"Created story world {sid}: {name}")

# If we have existing story worlds, allow selection
if st.session_state.story_worlds:
    st.sidebar.header("Your Story Worlds")
    choices = {f"{v['id']}: {v['name']}": v['id'] for v in st.session_state.story_worlds.values()}
    sel = st.sidebar.selectbox("Select story world", options=list(choices.keys()), index=0)
    st.session_state.story_id = choices[sel]

# --- Character creation ---
if st.session_state.story_id:
    st.subheader("Add a character to the current story world")
    with st.form("character_form"):
        cname = st.text_input("Character Name")
        ctraits = st.text_area("Traits (comma-separated or short sentences)")
        cback = st.text_area("Backstory (brief)")
        crole = st.text_input("Role (protagonist, antagonist, NPC, etc.)")
        created = st.form_submit_button("Create Character")
        if created:
            if not cname:
                st.warning("Character needs a name.")
            else:
                cid = create_character(st.session_state.story_id, cname, ctraits, cback, crole)
                st.success(f"Character {cid} created: {cname}")

# --- Interactive Story UI ---
if st.session_state.story_id:
    st.markdown("---")
    st.header("Interactive Story")
    sw = st.session_state.story_worlds[st.session_state.story_id]

    # Initialize first scene if none exists
    if not sw["scenes"]:
        initial_scene = "You wake up in a dimly lit room with no memory of how you arrived."
        sw["scenes"].append(initial_scene)
        index_scene_to_chroma(st.session_state.story_id, initial_scene)

    # Display the current story (last N scenes)
    with st.expander("Current story (recent)", expanded=True):
        for s in sw["scenes"][-10:]:
            st.write(s)

    # Show characters
    with st.expander("Characters", expanded=False):
        for c in sw["characters"].values():
            st.write(f"**{c['name']}** ({c['role']}) — {c['traits']}\n{c['backstory']}")

    # User input form for new action/choice
    with st.form("user_choice_form"):
        user_input = st.text_input("Your action / choice:")
        submit_choice = st.form_submit_button("Submit")
        if submit_choice and user_input:
            # 1) Generate next story continuation using RAG
            continuation = generate_response(st.session_state.story_id, user_input, model)
            st.success("Scene updated")

            # 2) Generate a follow-up question (also uses retrieval)
            question = generate_question(st.session_state.story_id, model)

            st.write("**Continuation:**")
            st.write(continuation)
            st.markdown("**Next question for the player:**")
            st.write(question)

    if st.button("Show top memories for last input"):
        last_input = sw["user_inputs"][-1] if sw["user_inputs"] else ""
        if last_input:
            retrieved = retrieve_relevant_docs(st.session_state.story_id, last_input)
            st.write("Retrieved memories:")
            for r in retrieved:
                st.write(r.get("text", str(r)))
        else:
            st.info("No user inputs yet to retrieve against.")

    st.info("All indexed vectors are stored persistently in the chroma_db folder next to this script.")

else:
    st.info("Create a story world to start.")