import streamlit as st
import tempfile

from DrownDetect import detectDrowning

# Page settings
st.set_page_config(
    page_title="Drowning Detection",
    layout="wide"
)
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    max-width: 1200px;
}

video {
    max-height: 350px !important;
}
</style>
""", unsafe_allow_html=True)

st.title("🏊 Drowning Detection System")

# DEMO VIDEOS

st.subheader("🎬 Try with a Demo Video")

demo_video = st.selectbox(
    "Choose a demo",
    [
        "Select Video",
        "drowning_001.mp4",
        "drowning_002.mp4",
        "test.mp4"
    ]
)

if demo_video != "Select Video":

    selected_video = f"videos/{demo_video}"

    col1, col2 = st.columns([2, 1])

    with col1:
        st.video(selected_video)

    with col2:

        st.info("Click Analyze to start detection")

        if st.button("Analyze Demo Video"):

            with st.spinner("Analyzing video..."):
                result = detectDrowning(selected_video)

            st.metric(
                "Drowning Frames",
                result["drowning_frames"]
            )

            st.metric(
                "Normal Frames",
                result["normal_frames"]
            )

            st.metric(
                "Risk Score",
                f"{result['risk_score']:.2f}%"
            )

            if result["is_drowning"]:
                st.error("⚠️ Drowning Detected")
            else:
                st.success("No Drowning Detected")
                st.divider()

st.subheader("📤 Upload Your Own Video")

uploaded_file = st.file_uploader(
    "Upload a video",
    type=["mp4", "avi", "mov"]
)

if uploaded_file:

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    )

    temp_file.write(uploaded_file.read())

    col1, col2 = st.columns([2, 1])

    with col1:
        st.video(temp_file.name)

    with col2:

        if st.button("Analyze Uploaded Video"):

            with st.spinner("Analyzing video..."):
                result = detectDrowning(temp_file.name)

            st.metric(
                "Drowning Frames",
                result["drowning_frames"]
            )

            st.metric(
                "Normal Frames",
                result["normal_frames"]
            )

            st.metric(
                "Risk Score",
                f"{result['risk_score']:.2f}%"
            )

            if result["is_drowning"]:
                st.error("⚠️ Drowning Detected")
            else:
                st.success("No Drowning Detected")

              