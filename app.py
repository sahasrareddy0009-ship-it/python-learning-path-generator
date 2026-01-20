import streamlit as st
from backend.generator import generate_learning_path

st.set_page_config(page_title="Python Learning Path Generator")

st.title("Python Learning Path Generator")
st.caption("AI-style personalized learning roadmap")

# ---------------- User Inputs ----------------
level = st.radio(
    "Select your Python level",
    ["Beginner", "Intermediate", "Advanced"],
    horizontal=True
)

goal = st.selectbox(
    "Select your learning objective",
    [
        "AI / Machine Learning",
        "Data Science",
        "Backend Development",
        "Full Stack Development",
        "Automation & Scripting",
        "Interview Preparation",
        "Competitive Programming",
        "Research & Advanced Python"
    ]
)

# ---------------- Generate ----------------
if st.button("Generate Learning Path"):
   st.markdown(path)


    st.divider()
    st.subheader("Your Personalized Learning Path")

    for phase, content in path.items():
        st.markdown(f"### {phase}")

        for topic in content["topics"]:
            st.markdown(f"- {topic}")

        with st.expander("Why this phase?"):
            st.write(content["why"])
