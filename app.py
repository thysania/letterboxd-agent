import os
import re
import sqlite3
from datetime import datetime
from typing import List, Dict

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

DB_PATH = "letterboxd_agent.db"

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-5.6-luna")

if BASE_URL:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
else:
    client = OpenAI(api_key=API_KEY)


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Personal Letterboxd Agent",
    page_icon="🎬",
    layout="wide",
)


# ---------------------------------------------------------
# Database
# ---------------------------------------------------------

def get_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database():
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS examples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            film_title TEXT NOT NULL,
            film_year TEXT,
            rating REAL,
            genres TEXT,
            tone TEXT,
            review TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS generations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            film_title TEXT NOT NULL,
            rating REAL,
            thoughts TEXT,
            tone TEXT,
            length TEXT,
            draft TEXT NOT NULL,
            final_review TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )

    connection.commit()
    connection.close()


def add_example(
    film_title: str,
    film_year: str,
    rating: float,
    genres: str,
    tone: str,
    review: str,
):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO examples
        (film_title, film_year, rating, genres, tone, review, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            film_title.strip(),
            film_year.strip(),
            rating,
            genres.strip(),
            tone.strip(),
            review.strip(),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    connection.commit()
    connection.close()


def get_examples() -> List[Dict]:
    connection = get_connection()
    rows = connection.execute(
        "SELECT * FROM examples ORDER BY id DESC"
    ).fetchall()
    connection.close()
    return [dict(row) for row in rows]


def delete_example(example_id: int):
    connection = get_connection()
    connection.execute("DELETE FROM examples WHERE id = ?", (example_id,))
    connection.commit()
    connection.close()


def save_generation(
    film_title: str,
    rating: float,
    thoughts: str,
    tone: str,
    length: str,
    draft: str,
    final_review: str = "",
):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO generations
        (film_title, rating, thoughts, tone, length, draft, final_review, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            film_title.strip(),
            rating,
            thoughts.strip(),
            tone,
            length,
            draft.strip(),
            final_review.strip(),
            datetime.now().isoformat(timespec="seconds"),
        ),
    )
    connection.commit()
    connection.close()


def get_generations() -> List[Dict]:
    connection = get_connection()
    rows = connection.execute(
        "SELECT * FROM generations ORDER BY id DESC"
    ).fetchall()
    connection.close()
    return [dict(row) for row in rows]


def get_setting(key: str, default: str = "") -> str:
    connection = get_connection()
    row = connection.execute(
        "SELECT value FROM settings WHERE key = ?",
        (key,),
    ).fetchone()
    connection.close()

    if row is None:
        return default

    return row["value"]


def save_setting(key: str, value: str):
    connection = get_connection()
    connection.execute(
        """
        INSERT INTO settings (key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    connection.commit()
    connection.close()


initialize_database()


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def word_count(text: str) -> int:
    return len(re.findall(r"\b[\w’'-]+\b", text))


def trim_text(text: str, max_chars: int = 3500) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[truncated]"


def normalize_rating(value) -> str:
    if value is None:
        return "unrated"

    try:
        return f"{float(value):.1f}/5"
    except Exception:
        return "unrated"


def choose_relevant_examples(
    examples: List[Dict],
    rating: float,
    tone: str,
    genres: str,
    limit: int = 6,
) -> List[Dict]:
    """
    Simple metadata-based retrieval.
    This is intentionally used instead of embeddings for the first version.
    """

    requested_genres = {
        item.strip().lower()
        for item in genres.split(",")
        if item.strip()
    }

    scored = []

    for example in examples:
        score = 0

        example_tone = (example.get("tone") or "").lower()
        example_genres = {
            item.strip().lower()
            for item in (example.get("genres") or "").split(",")
            if item.strip()
        }

        if tone and tone.lower() in example_tone:
            score += 4

        if requested_genres and requested_genres.intersection(example_genres):
            score += 3

        if example.get("rating") is not None:
            try:
                difference = abs(float(example["rating"]) - float(rating))
                if difference <= 0.5:
                    score += 3
                elif difference <= 1.0:
                    score += 1
            except Exception:
                pass

        # Prefer shorter examples when generating short reviews.
        review_length = word_count(example.get("review", ""))
        if review_length <= 150:
            score += 1

        scored.append((score, example))

    scored.sort(key=lambda item: item[0], reverse=True)

    selected = [example for _, example in scored[:limit]]

    # If metadata did not produce enough useful results, fill the remainder.
    if len(selected) < min(limit, len(examples)):
        selected_ids = {example["id"] for example in selected}

        for example in examples:
            if example["id"] not in selected_ids:
                selected.append(example)

            if len(selected) >= limit:
                break

    return selected[:limit]


def format_examples(examples: List[Dict]) -> str:
    if not examples:
        return "No reference examples are available."

    sections = []

    for index, example in enumerate(examples, start=1):
        metadata = [
            f"Film: {example.get('film_title', '')}",
            f"Rating: {normalize_rating(example.get('rating'))}",
        ]

        if example.get("film_year"):
            metadata.append(f"Year: {example['film_year']}")

        if example.get("genres"):
            metadata.append(f"Genres: {example['genres']}")

        if example.get("tone"):
            metadata.append(f"Tone tags: {example['tone']}")

        sections.append(
            f"REFERENCE REVIEW {index}\n"
            + "\n".join(metadata)
            + f"\nText:\n{trim_text(example.get('review', ''))}"
        )

    return "\n\n".join(sections)


# ---------------------------------------------------------
# Model functions
# ---------------------------------------------------------

def call_model(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.85,
) -> str:
    if not API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to your .env file."
        )

    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
    )

    content = response.choices[0].message.content

    if not content:
        raise RuntimeError("The model returned an empty response.")

    return content.strip()


def create_style_profile(examples: List[Dict]) -> str:
    if not examples:
        return (
            "No style profile has been created yet. "
            "Use a direct, conversational, specific movie-review voice."
        )

    review_text = format_examples(examples)

    system_prompt = """
You are a writing-style analyst.

Analyze the supplied movie reviews and create a practical style profile.
Describe observable patterns, not the identity of the writer.

Do not quote long passages.
Do not reproduce the reviews.
Do not tell another model to impersonate a living writer.
Focus on:
- tone
- humor
- sentence rhythm
- vocabulary
- review length
- structure
- opening and closing habits
- level of film analysis
- use of plot summary
- use of slang, profanity, lowercase, emojis, or punctuation
- recurring weaknesses to avoid

Return a concise profile that another model can use to write original reviews.
"""

    user_prompt = f"""
Here are the reference reviews:

{review_text}

Create the style profile.
"""

    return call_model(system_prompt, user_prompt, temperature=0.3)


def generate_reviews(
    film_title: str,
    film_year: str,
    rating: float,
    genres: str,
    thoughts: str,
    tone: str,
    length: str,
    style_profile: str,
    examples: List[Dict],
) -> List[str]:
    references = format_examples(examples)

    system_prompt = """
You write original personal movie reviews.

The reference material is provided to learn broad stylistic characteristics,
such as tone, rhythm, structure, level of detail, humor, and typical length.

Important rules:
- Do not copy sentences, phrases, jokes, metaphors, or distinctive wording.
- Do not imitate a living writer so closely that the result could be mistaken
  for their work.
- Do not invent facts about the film.
- Use the user's own thoughts as the source of opinion.
- Do not write a generic review full of empty praise.
- Do not summarize the entire plot unless the user asks for it.
- If the user's notes are limited, stay modest and avoid unsupported details.
- Make the result feel like a casual Letterboxd review rather than an essay.
"""

    user_prompt = f"""
STYLE PROFILE:
{style_profile}

REFERENCE REVIEWS:
{references}

NEW REVIEW REQUEST:
Film title: {film_title}
Film year: {film_year or "unknown"}
User rating: {rating}/5
Genres or tags: {genres or "not provided"}
Requested tone: {tone}
Requested length: {length}

The user's personal thoughts:
{thoughts}

Generate exactly three distinct original review drafts.

Label them exactly as:
DRAFT 1:
DRAFT 2:
DRAFT 3:

Each draft should follow the requested length and tone.
"""

    raw_output = call_model(system_prompt, user_prompt, temperature=0.9)

    drafts = re.split(
        r"(?i)DRAFT\s*[123]\s*:",
        raw_output,
    )

    drafts = [draft.strip() for draft in drafts if draft.strip()]

    if len(drafts) >= 3:
        return drafts[:3]

    # Fallback if the model did not follow the labels.
    paragraphs = [
        paragraph.strip()
        for paragraph in raw_output.split("\n\n")
        if paragraph.strip()
    ]

    if len(paragraphs) >= 3:
        return paragraphs[:3]

    return [raw_output]


def refine_review(
    draft: str,
    instruction: str,
    film_title: str,
    style_profile: str,
) -> str:
    system_prompt = """
You are an editor for a personal movie-review writing tool.

Revise the supplied draft according to the user's instruction.
Preserve the strongest specific idea and keep the voice natural.

Do not add unsupported film facts.
Do not copy the reference material.
Do not explain your changes.
Return only the revised review.
"""

    user_prompt = f"""
Film: {film_title}

Style profile:
{style_profile}

Current draft:
{draft}

Revision instruction:
{instruction}

Return only the revised review.
"""

    return call_model(system_prompt, user_prompt, temperature=0.75)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "drafts" not in st.session_state:
    st.session_state.drafts = []

if "selected_draft" not in st.session_state:
    st.session_state.selected_draft = 0

if "style_profile" not in st.session_state:
    st.session_state.style_profile = get_setting("style_profile", "")


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.title("🎬 Letterboxd Agent")

    st.caption(
        "A local writing assistant based on your own review examples."
    )

    st.write(f"Model: `{MODEL_NAME}`")

    if API_KEY:
        st.success("API key loaded")
    else:
        st.error("API key missing")

    examples = get_examples()

    st.metric("Reference reviews", len(examples))

    st.divider()

    st.subheader("Style profile")

    if st.session_state.style_profile:
        st.text_area(
            "Current profile",
            value=st.session_state.style_profile,
            height=250,
            key="style_profile_editor",
        )

        if st.button("Save edited profile", use_container_width=True):
            st.session_state.style_profile = st.session_state.style_profile_editor
            save_setting("style_profile", st.session_state.style_profile)
            st.success("Style profile saved.")
    else:
        st.info("Add examples, then generate a style profile.")


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------

st.title("Personal Movie Review Generator")
st.write(
    "Add reviews you like, describe your reaction to a film, and generate "
    "original drafts using the patterns from your reference collection."
)

tab_generate, tab_examples, tab_history, tab_settings = st.tabs(
    [
        "✍️ Generate",
        "📚 Reference reviews",
        "🗂️ History",
        "⚙️ Settings",
    ]
)


# ---------------------------------------------------------
# Generate tab
# ---------------------------------------------------------

with tab_generate:
    st.subheader("Generate a review")

    if not examples:
        st.warning(
            "Add at least one reference review before generating. "
            "Three to ten examples is a good starting point."
        )

    left, right = st.columns([1, 1])

    with left:
        film_title = st.text_input(
            "Film title",
            placeholder="The Shining",
        )

        film_year = st.text_input(
            "Year",
            placeholder="1980",
        )

        rating = st.slider(
            "Your rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0,
            step=0.5,
        )

        genres = st.text_input(
            "Genres or tags",
            placeholder="horror, psychological, Kubrick",
        )

    with right:
        tone = st.selectbox(
            "Desired tone",
            [
                "closest to my normal voice",
                "funny",
                "dry",
                "sincere",
                "enthusiastic",
                "mixed or conflicted",
                "harsh",
                "analytical",
                "poetic",
            ],
        )

        length = st.selectbox(
            "Length",
            [
                "one sentence",
                "very short, around 25–50 words",
                "short, around 50–100 words",
                "medium, around 100–180 words",
                "long, around 180–300 words",
            ],
        )

        st.write("Your thoughts")

        thoughts = st.text_area(
            "Your thoughts",
            height=170,
            placeholder=(
                "What did you like? What annoyed you? Favorite scene? "
                "How did the ending land? Avoiding spoilers?"
            ),
            label_visibility="collapsed",
        )

    generate_button = st.button(
        "Generate three drafts",
        type="primary",
        use_container_width=True,
    )

    if generate_button:
        if not API_KEY:
            st.error("Add OPENAI_API_KEY to your .env file first.")
        elif not film_title.strip():
            st.error("Enter a film title.")
        elif not thoughts.strip():
            st.error(
                "Add your own thoughts first. The more specific they are, "
                "the less generic the result will be."
            )
        elif not examples:
            st.error("Add at least one reference review first.")
        else:
            with st.spinner("Generating drafts..."):
                try:
                    relevant_examples = choose_relevant_examples(
                        examples=examples,
                        rating=rating,
                        tone=tone,
                        genres=genres,
                        limit=6,
                    )

                    if not st.session_state.style_profile:
                        st.session_state.style_profile = create_style_profile(
                            examples
                        )
                        save_setting(
                            "style_profile",
                            st.session_state.style_profile,
                        )

                    drafts = generate_reviews(
                        film_title=film_title,
                        film_year=film_year,
                        rating=rating,
                        genres=genres,
                        thoughts=thoughts,
                        tone=tone,
                        length=length,
                        style_profile=st.session_state.style_profile,
                        examples=relevant_examples,
                    )

                    st.session_state.drafts = drafts
                    st.session_state.selected_draft = 0

                    st.success("Drafts generated.")

                except Exception as error:
                    st.error(f"Generation failed: {error}")

    if st.session_state.drafts:
        st.divider()
        
