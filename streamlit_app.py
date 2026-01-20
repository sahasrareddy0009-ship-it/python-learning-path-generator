"""
Streamlit UI for RAG-Based Python Learning Path Generator

This is the main application file that provides an interactive web interface
for generating personalized Python learning paths.

Run with: streamlit run app/streamlit_app.py
"""

import streamlit as st
import yaml
# import sys
# from pathlib import Path
from dataclasses import asdict
import time

# Add src directory to path for imports - must be at the top
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
# sys.path.insert(0, str(SRC_DIR))

# Import our RAG modules
from retriever import (
    UserProfile,
    RuleBasedRetriever,
    KnowledgeBaseLoader,
    generate_retrieval,
)
from generator import GroqGenerator, create_generator
from explainer import ExplainableAI, create_explainer

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Python Learning Path Generator",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS - BLACK BACKGROUND, WHITE TEXT ONLY
# =============================================================================

st.markdown(
    """
<style>
/* Main background - black */
.stApp {
    background-color: #000000;
    color: #FFFFFF;
}

/* All text white */
p, li, span, div {
    color: #FFFFFF !important;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #FFFFFF !important;
    font-weight: 700;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0A0A0A;
    border-right: 1px solid #333333;
}

/* Sidebar headers */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
[data-testid="stSidebar"] h4 {
    color: #FFFFFF !important;
}

/* Custom highlight box - black with white border */
.highlight-box {
    background-color: #0A0A0A;
    border: 1px solid #FFFFFF;
    border-radius: 4px;
    padding: 20px;
    margin: 16px 0;
}

.highlight-box h4 {
    color: #FFFFFF !important;
    margin-top: 0 !important;
}

.highlight-box p {
    color: #FFFFFF !important;
    font-size: 15px;
    line-height: 1.6;
}

/* Success box */
.success-box {
    background-color: #0A0A0A;
    border: 1px solid #FFFFFF;
    border-radius: 4px;
    padding: 20px;
    margin: 16px 0;
}

.success-box h4 {
    color: #FFFFFF !important;
    margin-top: 0 !important;
}

.success-box p {
    color: #FFFFFF !important;
    font-size: 15px;
    line-height: 1.6;
}

/* Info box - more visible */
.info-box {
    background-color: #151515;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 14px;
    margin: 10px 0;
}

.info-box p {
    color: #FFFFFF !important;
    font-size: 14px;
    line-height: 1.5;
    margin: 0 !important;
}

/* Topic card - more visible */
.topic-card {
    background-color: #151515;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 18px;
    margin: 14px 0;
}

.topic-card h4 {
    color: #FFFFFF !important;
    margin-top: 0 !important;
    margin-bottom: 10px !important;
    font-size: 16px;
}

.topic-card p {
    color: #FFFFFF !important;
    font-size: 14px;
    line-height: 1.5;
    margin: 6px 0 !important;
}

/* Metric card */
.metric-card {
    background-color: #151515;
    border: 1px solid #444444;
    border-radius: 4px;
    padding: 24px 16px;
    text-align: center;
}

.metric-card h3 {
    font-size: 32px;
    margin: 0 !important;
    color: #FFFFFF !important;
}

.metric-card p {
    color: #CCCCCC !important;
    font-size: 13px;
    margin: 6px 0 0 0 !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background-color: #0A0A0A;
    padding: 6px;
    border-radius: 4px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #1A1A1A;
    border-radius: 4px;
    padding: 10px 18px;
    color: #FFFFFF !important;
    font-size: 14px;
}

.stTabs [aria-selected="true"] {
    background-color: #333333 !important;
    color: #FFFFFF !important;
}

/* TABS CONTENT - ENSURE WHITE TEXT AND BLACK BACKGROUND */
.stTabs [data-baseweb="tab-panel"] {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    padding: 20px;
}

.stTabs [data-baseweb="tab-panel"] p {
    color: #FFFFFF !important;
}

.stTabs [data-baseweb="tab-panel"] li {
    color: #FFFFFF !important;
}

.stTabs [data-baseweb="tab-panel"] h1,
.stTabs [data-baseweb="tab-panel"] h2,
.stTabs [data-baseweb="tab-panel"] h3,
.stTabs [data-baseweb="tab-panel"] h4 {
    color: #FFFFFF !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background-color: #151515;
    border-radius: 4px;
    color: #FFFFFF !important;
    font-weight: 600;
    font-size: 14px;
    padding: 12px 16px;
}

.streamlit-expanderHeader:hover {
    background-color: #222222 !important;
}

/* HIDE EXPANDER LINK ICON - Remove deep-link functionality */
.streamlit-expanderHeader a,
.streamlit-expanderHeader button[data-testid="expanderToggleIcon"],
[data-testid="stExpander"] a,
[data-testid="stExpander"] button {
    display: none !important;
}

/* Hide the expander's native toggle icon if unwanted */
.streamlit-expanderHeader svg {
    display: none !important;
}

/* EXPANDER CONTENT - BLACK BACKGROUND WITH WHITE TEXT */
.streamlit-expanderContent {
    background-color: #000000 !important;
    color: #FFFFFF !important;
    padding: 16px;
    border: 1px solid #333333;
    border-top: none;
    border-radius: 0 0 4px 4px;
}

.streamlit-expanderContent p {
    color: #FFFFFF !important;
}

.streamlit-expanderContent li {
    color: #FFFFFF !important;
}

/* Buttons - HIGH VISIBILITY */
.stButton > button {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    border-radius: 8px;
    padding: 20px 40px;
    font-weight: 700;
    font-size: 18px;
    border: 3px solid #FFFFFF !important;
    width: 100%;
    margin-top: 30px;
    margin-bottom: 30px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
}

.stButton > button:hover {
    background-color: #EEEEEE !important;
    border-color: #FFFFFF !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

/* Sidebar button - HIGH CONTRAST */
[data-testid="stSidebar"] .stButton > button {
    background-color: #FFFFFF !important;
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border: 2px solid #FFFFFF !important;
    border-radius: 8px !important;
    padding: 16px 24px !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #EEEEEE !important;
    color: #000000 !important;
}

/* Force button text visibility */
button[kind="primary"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* Select boxes - more visible */
[data-baseweb="select"] > div {
    background-color: #151515;
    border-radius: 4px;
    border: 1px solid #444444;
}

[data-baseweb="select"] * {
    color: #FFFFFF !important;
}

/* Info messages */
[data-testid="stInfo"] {
    background-color: #151515;
    border: 1px solid #444444;
    border-radius: 4px;
    color: #FFFFFF !important;
}

/* Success messages */
[data-testid="stSuccess"] {
    background-color: #151515;
    border: 1px solid #444444;
    border-radius: 4px;
    color: #FFFFFF !important;
}

/* Warning messages */
[data-testid="stWarning"] {
    background-color: #151515;
    border: 1px solid #444444;
    border-radius: 4px;
    color: #FFFFFF !important;
}

/* Divider */
hr {
    border-color: #333333;
    margin: 20px 0;
}

/* Progress bar */
[data-testid="stProgress"] > div > div {
    background-color: #FFFFFF;
}

/* Slider */
[data-baseweb="slider"] {
    color: #FFFFFF;
}

/* Input fields */
[data-baseweb="input"] {
    background-color: #151515;
    border: 1px solid #444444;
}

[data-baseweb="input"] * {
    color: #FFFFFF !important;
}

/* Radio buttons */
[data-testid="stRadio"] label {
    color: #FFFFFF !important;
}

/* Checkbox */
[data-testid="stCheckbox"] label {
    color: #FFFFFF !important;
}

/* Hide Streamlit footer */
footer {
    visibility: hidden;
}

/* Make all borders visible */
* {
    border-color: #333333 !important;
}

/* Spinner */
[data-testid="stSpinner"] {
    color: #FFFFFF !important;
}

/* ALL TEXT IN MAIN AREA - WHITE ON BLACK */
.main .block-container {
    background-color: #000000 !important;
    color: #FFFFFF !important;
}

.main .block-container p {
    color: #FFFFFF !important;
}

.main .block-container li {
    color: #FFFFFF !important;
}

.main .block-container h1,
.main .block-container h2,
.main .block-container h3,
.main .block-container h4 {
    color: #FFFFFF !important;
}

/* MARKDOWN CONTENT */
.stMarkdown {
    color: #FFFFFF !important;
}

.stMarkdown p {
    color: #FFFFFF !important;
}

.stMarkdown li {
    color: #FFFFFF !important;
}

.stMarkdown h1,
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4 {
    color: #FFFFFF !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


@st.cache_resource
def load_knowledge_base():
    """Load knowledge base (cached for performance)"""
    kb_path = PROJECT_ROOT / "knowledge_base" / "python_topics.yaml"
    return KnowledgeBaseLoader(str(kb_path))


@st.cache_resource
def load_retriever():
    """Load retriever (cached for performance)"""
    kb_path = PROJECT_ROOT / "knowledge_base" / "python_topics.yaml"
    return RuleBasedRetriever(str(kb_path))


@st.cache_resource
def load_explainer():
    """Load explainer (cached for performance)"""
    kb_path = PROJECT_ROOT / "knowledge_base" / "python_topics.yaml"
    return create_explainer(str(kb_path))


def get_career_goal_info(goal_id: str) -> dict:
    """Get information about a career goal"""
    kb = load_knowledge_base().load()
    return kb.get("career_tracks", {}).get(goal_id, {})


def get_learning_style_info(style_id: str) -> dict:
    """Get information about a learning style"""
    kb = load_knowledge_base().load()
    return kb.get("learning_styles", {}).get(style_id, {})


def get_career_objective_info(objective_id: str) -> dict:
    """Get information about a career objective"""
    kb = load_knowledge_base().load()
    return kb.get("career_objectives", {}).get(objective_id, {})


def generate_learning_path(user_profile: UserProfile, progress_bar=None):
    """
    Generate a learning path using the RAG pipeline.
    """
    # Step 1: Retrieve relevant topics
    retriever = load_retriever()
    retrieved_topics, explanations = retriever.retrieve(user_profile, max_topics=15)

    if progress_bar:
        progress_bar.progress(40)

    # Step 2: Generate personalized path using LLM
    generator = create_generator(provider="groq")

    style_config = get_learning_style_info(user_profile.learning_style)

    generated_path = generator.generate_path(
        retrieved_topics=retrieved_topics,
        user_profile=asdict(user_profile),
        explanations=explanations,
        style_config=style_config,
    )

    if progress_bar:
        progress_bar.progress(70)

    # Step 3: Generate explanations
    explainer = load_explainer()
    path_explanation = explainer.explain_path(
        retrieved_topics, asdict(user_profile), {}
    )

    if progress_bar:
        progress_bar.progress(100)

    return {
        "retrieved_topics": retrieved_topics,
        "generated_path": generated_path,
        "path_explanation": path_explanation,
        "explanations": explanations,
    }


# =============================================================================
# SIDEBAR - USER INPUT FORM
# =============================================================================


def render_sidebar():
    """Render the sidebar with user input form"""
    st.sidebar.markdown(
        """
<div style="padding: 10px 0 15px 0;">
    <h2 style="font-size: 20px;">Your Learning Profile</h2>
</div>
<hr style="margin: 12px 0;">
""",
        unsafe_allow_html=True,
    )

    # Career Goal Selection
    st.sidebar.markdown("**1. Career Goal**")

    career_goals = [
        ("web_development", "Web Development"),
        ("data_science", "Data Science & Analytics"),
        ("machine_learning", "Machine Learning & AI"),
        ("automation_scripting", "Automation & Scripting"),
        ("backend_api", "Backend API Development"),
        ("scientific_computing", "Scientific Computing"),
    ]

    career_goal = st.sidebar.selectbox(
        "Select career goal",
        options=[g[0] for g in career_goals],
        format_func=lambda x: next((g[1] for g in career_goals if g[0] == x), x),
        index=1,
        label_visibility="visible",
    )

    # Show career goal info
    goal_info = get_career_goal_info(career_goal)
    if goal_info:
        st.sidebar.markdown(
            f"""
    <div class="info-box">
    <b>{goal_info.get("name", "")}</b><br>
    {goal_info.get("description", "")}
    </div>
    """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)

    # Current Skill Level
    st.sidebar.markdown("**2. Current Skill Level**")

    skill_levels = [
        ("absolute_beginner", "Absolute Beginner"),
        ("beginner", "Beginner"),
        ("intermediate", "Intermediate"),
        ("advanced", "Advanced"),
    ]

    skill_level = st.sidebar.selectbox(
        "Select skill level",
        options=[s[0] for s in skill_levels],
        format_func=lambda x: next((s[1] for s in skill_levels if s[0] == x), x),
        index=1,
        label_visibility="visible",
    )

    st.sidebar.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)

    # Learning Style
    st.sidebar.markdown("**3. Learning Style**")

    learning_styles = [
        ("project_based", "Project-Based"),
        ("theory_first", "Theory-First"),
        ("balanced", "Balanced"),
        ("challenge_driven", "Challenge-Driven"),
    ]

    learning_style = st.sidebar.selectbox(
        "Select learning style",
        options=[l[0] for l in learning_styles],
        format_func=lambda x: next((l[1] for l in learning_styles if l[0] == x), x),
        index=0,
        label_visibility="visible",
    )

    # Show learning style info
    style_info = get_learning_style_info(learning_style)
    if style_info:
        st.sidebar.markdown(
            f"""
    <div class="info-box">
    <b>Approach:</b> {style_info.get("approach", "")}
    </div>
    """,
            unsafe_allow_html=True,
        )

    st.sidebar.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)

    # Time Availability
    st.sidebar.markdown("**4. Weekly Time Commitment**")

    time_options = [
        "5 hours/week",
        "10 hours/week",
        "15 hours/week",
        "20+ hours/week",
    ]

    time_availability = st.sidebar.select_slider(
        "Select weekly hours", options=time_options, value="10 hours/week"
    )

    st.sidebar.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)

    # Career Objective
    st.sidebar.markdown("**5. Career Objective**")

    career_objectives = [
        ("junior_developer", "Junior Developer"),
        ("mid_level_developer", "Mid-Level Developer"),
        ("data_analyst", "Data Analyst"),
        ("data_scientist", "Data Scientist"),
        ("ml_engineer", "ML Engineer"),
    ]

    career_objective = st.sidebar.selectbox(
        "Select career objective",
        options=[c[0] for c in career_objectives],
        format_func=lambda x: next((c[1] for c in career_objectives if c[0] == x), x),
        index=0,
        label_visibility="visible",
    )

    st.sidebar.markdown("<hr style='margin: 16px 0;'>", unsafe_allow_html=True)

    # Generate Button - Form-based for instant updates
    st.sidebar.markdown("### Generate Your Path")
    st.sidebar.markdown("*Change any option above to update*")

    # Use a form that triggers only on button click
    with st.sidebar.form(key="learning_profile_form"):
        st.markdown(
            """
    <style>
        /* Make selectboxes not editable - show only dropdown arrow */
        [data-baseweb="select"] {
            cursor: default !important;
        }
        [data-baseweb="select"] > div {
            cursor: default !important;
            background-color: #151515 !important;
        }
        [data-baseweb="select"] input {
            cursor: default !important;
            pointer-events: none !important;
        }
        
        /* Form button styling */
        .form-button .stButton > button {
            width: 100%;
            background-color: #FFFFFF !important;
            color: #000000 !important;
            font-weight: 700 !important;
            font-size: 16px !important;
            padding: 16px 24px !important;
            border: 3px solid #FFFFFF !important;
            border-radius: 8px !important;
            cursor: pointer !important;
        }
        .form-button .stButton > button:hover {
            background-color: #EEEEEE !important;
        }
        
        /* Disable cursor pointer on selectboxes */
        .stSelectbox {
            cursor: default !important;
        }
    </style>
    """,
            unsafe_allow_html=True,
        )

        submit_btn = st.form_submit_button(
            "Generate My Learning Path",
            type="primary",
            use_container_width=True,
        )

        if submit_btn:
            st.session_state.generate_path = True
            st.session_state.last_profile = {
                "career_goal": career_goal,
                "skill_level": skill_level,
                "learning_style": learning_style,
                "time_availability": time_availability,
                "career_objective": career_objective,
            }

    # Initialize session state for generation
    if "generate_path" not in st.session_state:
        st.session_state.generate_path = False
    if "last_profile" not in st.session_state:
        st.session_state.last_profile = {}

    # Generate ONLY when button is clicked
    generate_btn = st.session_state.generate_path

    # Show API key input if needed
    with st.sidebar.expander("API Configuration", expanded=False):
        api_key = st.text_input("Groq API Key", type="password")
        if api_key:
            import os

            os.environ["GROQ_API_KEY"] = api_key
            st.success("API key set!")

    return {
        "career_goal": career_goal,
        "skill_level": skill_level,
        "learning_style": learning_style,
        "time_availability": time_availability,
        "career_objective": career_objective,
        "generate_btn": generate_btn,
    }


# =============================================================================
# MAIN CONTENT - PATH DISPLAY
# =============================================================================


def render_main_header():
    """Render the main header"""
    st.markdown(
        """
<div style="text-align: center; padding: 40px 25px; border: 1px solid #333333; border-radius: 6px; margin-bottom: 30px;">
    <h1 style="font-size: 36px; margin-bottom: 12px;">Python Learning Path Generator</h1>
    <p style="font-size: 18px; color: #AAAAAA !important;">Personalized, goal-oriented roadmaps powered by AI</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_path_result(
    result: dict, hours_per_week: int = 10, user_profile: dict | None = None
):
    """Render the generated learning path"""
    generated_path = result["generated_path"]
    path_explanation = result["path_explanation"]
    retrieved_topics = result["retrieved_topics"]

    # Calculate total hours from all topics (handle both int and string formats)
    def parse_hours(hours_val):
        if isinstance(hours_val, (int, float)):
            return float(hours_val)
        try:
            parts = str(hours_val).split("-")
            return (float(parts[0]) + float(parts[1])) / 2
        except (ValueError, IndexError):
            return 10.0  # Default to 10 hours

    total_hours = sum(parse_hours(rt.topic.duration_hours) for rt in retrieved_topics)

    # Calculate estimated weeks based on user's weekly time commitment
    if hours_per_week > 0:
        estimated_weeks = max(1, round(total_hours / hours_per_week))
    else:
        estimated_weeks = int(total_hours)  # Assume 1 hour per week if not specified

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["Your Learning Path", "Topic Details", "Why This Path?", "Projects"]
    )

    with tab1:
        render_learning_path_view(
            generated_path,
            path_explanation,
            hours_per_week,
            estimated_weeks,
            user_profile,
        )

    with tab2:
        render_topic_details_view(retrieved_topics, path_explanation)

    with tab3:
        render_explanation_view(path_explanation)

    with tab4:
        render_projects_view(generated_path)


def render_learning_path_view(
    generated_path,
    path_explanation,
    hours_per_week: int = 10,
    calculated_weeks: int | None = None,
    user_profile: dict | None = None,
):
    """Render the main learning path view"""
    # Use calculated weeks if provided, otherwise use the generator's estimate
    display_weeks = (
        calculated_weeks
        if calculated_weeks
        else int(generated_path.total_weeks.split("-")[0])
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
    <div class="metric-card">
        <h3>{generated_path.total_topics}</h3>
        <p>Topics</p>
    </div>
    """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
    <div class="metric-card">
        <h3>{display_weeks}</h3>
        <p>Weeks ({hours_per_week}h/week)</p>
    </div>
    """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
    <div class="metric-card">
        <h3>{len(generated_path.sections)}</h3>
        <p>Phases</p>
    </div>
    """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
    <div class="metric-card">
        <h3>{len(generated_path.project_suggestions)}</h3>
        <p>Projects</p>
    </div>
    """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Personalized header for this specific user
    career_goal = (
        user_profile.get("career_goal", "Python Development").replace("_", " ")
        if user_profile
        else "Python Development"
    )
    skill_level = (
        user_profile.get("skill_level", "beginner").replace("_", " ")
        if user_profile
        else "beginner"
    )
    career_objective = (
        user_profile.get("career_objective", "Python Developer").replace("_", " ")
        if user_profile
        else "Python Developer"
    )

    st.markdown(
        f"""
<div class="success-box" style="border-left: 4px solid #FFFFFF;">
<h4 style="margin-top: 0;">Your Personalized Path for {career_goal.title()}</h4>
<p>Based on your <b>{skill_level.title()}</b> level and goal to become a <b>{career_objective.title()}</b>, 
we've created a custom learning journey with {generated_path.total_topics} topics across {len(generated_path.sections)} phases.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    # Career outcome
    if generated_path.career_outcome:
        st.info(f"**Career Outcome:** {generated_path.career_outcome}")

    st.markdown("---")

    # Render sections
    for i, section in enumerate(generated_path.sections, 1):
        # Calculate section hours (sum of all topics in section)
        section_hours = sum(
            len(topic.get("key_concepts", []))
            * 2  # Rough estimate: 2 hours per concept
            for topic in section.topics
        )
        # Calculate weeks for this section based on hours_per_week
        section_weeks = (
            max(1, round(section_hours / hours_per_week))
            if section_hours > 0
            else section.estimated_weeks
        )

        with st.expander(
            f"Phase {i}: {section.title} ({section_weeks} weeks @ {hours_per_week}h/week)",
            expanded=True,
        ):
            st.markdown(f"**{section.description}**")

            # Topics in this section
            st.markdown("**Topics Covered**")
            for j, topic in enumerate(section.topics, 1):
                st.markdown(
                    f"""
                <div class="topic-card">
                <h4>{j}. {topic.get("name", "")}</h4>
                <p><b>Why included:</b> {topic.get("why_included", "")}</p>
                <p><b>Key concepts:</b> {", ".join(topic.get("key_concepts", [])[:3])}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Milestones
            if section.milestones:
                st.markdown("**Milestones**")
                for milestone in section.milestones:
                    st.markdown(f"- {milestone}")


def render_topic_details_view(retrieved_topics, path_explanation):
    """Render detailed topic information"""
    st.markdown("## All Topics in Your Path")

    for i, (topic_exp, rt) in enumerate(
        zip(path_explanation.topic_explanations, retrieved_topics), 1
    ):
        topic = rt.topic

        with st.expander(
            f"{i}. {topic_exp.topic_name} (Relevance: {topic_exp.relevance_score:.0%})",
            expanded=False,
        ):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"**Why included:** {topic_exp.primary_reason}")

                if topic_exp.contributing_factors:
                    st.markdown("**Contributing factors:**")
                    for factor in topic_exp.contributing_factors:
                        st.markdown(f"- {factor}")

                if topic_exp.career_relevance:
                    st.markdown(f"**Career relevance:** {topic_exp.career_relevance}")

                if topic_exp.skill_alignment:
                    st.markdown(f"**Skill alignment:** {topic_exp.skill_alignment}")

            with col2:
                st.markdown(
                    f"""
                <div class="info-box">
                <b>Level:</b> {topic.level.title()}<br>
                <b>Duration:</b> {topic.duration_hours} hours<br>
                <b>Importance:</b> {topic.importance.title()}
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Show concepts
            if topic.concepts:
                st.markdown("**Key Concepts:**")
                concepts_cols = st.columns(4)
                for j, concept in enumerate(topic.concepts):
                    concepts_cols[j % 4].markdown(f"- {concept}")


def render_explanation_view(path_explanation):
    """Render the explainable AI section"""
    st.markdown("## Why This Path? (Explainable AI)")

    # Overall approach
    st.markdown("### How This Path Was Generated")
    st.markdown(path_explanation.overall_approach)

    st.markdown("---")

    # Personalization factors
    st.markdown("### Personalization Factors")
    for factor, explanation in path_explanation.personalization_factors.items():
        st.markdown(f"**{factor.replace('_', ' ').title()}:** {explanation}")

    st.markdown("---")

    # Career outcome reasoning
    st.markdown("### Career Outcome")
    st.markdown(path_explanation.career_outcome_reasoning)

    st.markdown("---")

    # Timeline reasoning
    st.markdown("### Estimated Timeline")
    st.markdown(path_explanation.estimated_timeline_reasoning)

    st.markdown("---")

    # RAG Architecture explanation
    st.markdown("### How the RAG System Works")
    st.markdown("""
This learning path was generated using a **Retrieval-Augmented Generation (RAG)** pipeline:

1. **Retrieval:** Relevant topics were retrieved from the structured knowledge base based on your profile
2. **Reasoning:** Rule-based logic ensured proper prerequisites and topic ordering
3. **Generation:** An LLM synthesized the retrieved knowledge into a coherent, personalized path
4. **Explanation:** Transparent reasoning for each recommendation

Unlike template-based systems, this approach generates **unique, explainable** recommendations for every user.
""")


def render_projects_view(generated_path):
    """Render project suggestions"""
    st.markdown("## Suggested Projects")

    if not generated_path.project_suggestions:
        st.info("No project suggestions available for this path.")
        return

    for i, project in enumerate(generated_path.project_suggestions, 1):
        with st.expander(
            f"{i}. {project.get('name', 'Project')} ({project.get('difficulty', 'intermediate')})",
            expanded=True,
        ):
            st.markdown(
                f"**Why this project:** {project.get('why_this_project', 'Applied learning')}"
            )

            skills = project.get("skills_applied", [])
            if skills:
                st.markdown("**Skills Applied:**")
                for skill in skills:
                    st.markdown(f"- {skill}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================


def main():
    """Main application entry point"""
    # Render header
    render_main_header()

    # Render sidebar
    user_inputs = render_sidebar()

    # Check if user wants to generate (from form submission or profile change)
    if user_inputs["generate_btn"]:
        # Reset the generate flag so it doesn't regenerate on every change
        st.session_state.generate_path = False

        # Create user profile - fix for "20+ hours/week"
        time_str = user_inputs["time_availability"]
        # Extract numbers from string like "20+ hours/week" or "10 hours/week"
        hours_str = "".join(filter(str.isdigit, time_str.split()[0]))
        hours_per_week = int(hours_str) if hours_str else 10

        user_profile = UserProfile(
            skill_level=user_inputs["skill_level"],
            career_goal=user_inputs["career_goal"],
            learning_style=user_inputs["learning_style"],
            time_availability=user_inputs["time_availability"],
            career_objective=user_inputs["career_objective"],
            hours_per_week=hours_per_week,
        )

        # Show progress
        with st.spinner(
            "Analyzing your profile and generating your personalized path..."
        ):
            progress_bar = st.progress(0)

            try:
                # Generate the learning path
                result = generate_learning_path(user_profile, progress_bar)

                # Clear the progress bar
                time.sleep(0.3)
                progress_bar.empty()

                # Render results
                render_path_result(
                    result,
                    hours_per_week=hours_per_week,
                    user_profile=asdict(user_profile),
                )

            except Exception as e:
                progress_bar.empty()
                st.error(f"Error generating path: {str(e)}")
                st.info("Make sure you have set a valid Groq API key in the sidebar.")

    else:
        # Show welcome / empty state
        st.markdown(
            """
    <div class="highlight-box">
    <h4 style="margin-top: 0 !important;">Welcome to Your Python Learning Journey</h4>
    <p>Configure your learning profile in the sidebar, then click the button below to get your personalized Python learning roadmap.</p>
    
    <p><b>What makes this path special:</b></p>
    <ul>
        <li>Personalized - Based on your career goals and skill level</li>
        <li>Structured - Topics ordered with proper prerequisites</li>
        <li>Explainable - Clear reasoning for every recommendation</li>
        <li>Actionable - Includes milestones and project ideas</li>
    </ul>
    </div>
    """,
            unsafe_allow_html=True,
        )

        # Add a visible placeholder button that will be replaced when clicked
        st.markdown("### Ready to Start?")
        st.info(
            "Please configure your profile in the sidebar and click 'Generate My Learning Path'"
        )


if __name__ == "__main__":
    # Reset session state on reload
    if "generate_path" not in st.session_state:
        st.session_state.generate_path = False
    main()
