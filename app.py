import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Cat vs Dog AI Classifier",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>

/* Hide Streamlit Header */
header{
    visibility:hidden;
}

/* Remove Top Space */
.block-container{
    padding-top:0rem;
    padding-bottom:2rem;
    padding-left:2rem;
    padding-right:2rem;
}

/* Main Background */
.stApp{
    background:linear-gradient(135deg,#050816,#0B1120,#111827);
    color:white;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0F172A,#1E3A8A,#059669);
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Upload Box */
/* Upload Box */
[data-testid="stFileUploader"]{
    background:#1F2937 !important;
    border:2px solid #00FF99 !important;
    border-radius:15px !important;
    padding:15px !important;
}

/* Upload Label */
label[data-testid="stWidgetLabel"] p{
    color:white !important;
    font-size:22px !important;
    font-weight:bold !important;
}

/* Upload text */
[data-testid="stFileUploader"] span{
    color:black !important;
    font-size:18px !important;
}

/* Upload button */
[data-testid="stFileUploader"] button{
    background:grey!important;
    color:white !important;
    font-weight:bold !important;
}

label[data-testid="stWidgetLabel"] p{
    color:white !important;
    font-size:22px !important;
    font-weight:bold !important;
}

//* Force all text inside uploader */
[data-testid="stFileUploader"] *{
    color: black !important;
}

/* Metric Cards */
[data-testid="metric-container"]{
    background:#1F2937;
    border-radius:15px;
    border:2px solid #00E5FF;
    padding:10px;
}

/* Make metric text white */
[data-testid="metric-container"] label,
[data-testid="metric-container"] div,
[data-testid="metric-container"] p {
    color: white !important;
}

/* Make metric value white */
[data-testid="stMetricValue"] {
    color: white !important;
    font-size: 40px !important;
    font-weight: bold !important;
}

/* Make metric label white */
[data-testid="stMetricLabel"] {
    color: white !important;
    font-size: 18px !important;
}

/* Progress Bar */
.stProgress > div > div > div > div{
    background:#00FF99;
}

hr{
    border:1px solid #1E293B;
}
/* Increase Expander title size */
[data-testid="stExpander"] summary {
    font-size: 32px !important;
    font-weight: bold !important;
    color: #38BDF8 !important;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------

st.sidebar.title("🤖 About Project")

st.sidebar.markdown("""

## CNN Image Classification

📂 **Dataset:** Cats vs Dogs

🧠 **Algorithm:** CNN

⚙️ **Framework:** TensorFlow

🌐 **Frontend:** Streamlit

📏 **Image Size:** 128 × 128

🚀 **AI Powered Image Classification**

""")

# -----------------------------
# LOAD MODEL
# -----------------------------

model = tf.keras.models.load_model(
    "models/cnn_model.h5"
)

class_names = ["Cat","Dog"]

# -----------------------------
# TITLE
# -----------------------------

st.markdown("""
<h1 style="
text-align:center;
font-size:55px;
font-weight:bold;
">

🐱🐶
<span style="color:#38BDF8;">
Cat vs Dog
</span>

<span style="color:#22C55E;">
AI Classifier
</span>

</h1>
""",
unsafe_allow_html=True)

st.markdown("""
<h3 style="
text-align:center;
color:#CBD5E1;
">

🚀 Powered by Deep Learning (CNN)

</h3>
""",
unsafe_allow_html=True)

st.markdown("---")

# -----------------------------
# FILE UPLOADER
# -----------------------------

uploaded_file = st.file_uploader(
    "📂 Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    image = image.resize((128,128))

    img_array = np.array(
        image,
        dtype=np.float32
    )

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    prediction = model.predict(img_array)

    predicted = np.argmax(prediction)

    confidence = prediction[0][predicted]*100
        # -----------------------------
    # AI PREDICTION CARD
    # -----------------------------

    with col2:

        st.markdown("""
        <h2 style="
        text-align:center;
        color:#38BDF8;
        ">
        🤖 AI Prediction
        </h2>
        """,
        unsafe_allow_html=True)

        if predicted == 0:

            st.markdown(f"""
            <div style="
            background:#111827;
            border:3px solid #38BDF8;
            border-radius:20px;
            padding:25px;
            text-align:center;
            box-shadow:0px 0px 20px #38BDF8;
            ">

            <h1 style="color:#38BDF8;">
            🐱 CAT
            </h1>

            <h2 style="color:white;">
            {confidence:.2f}% Confidence
            </h2>

            </div>
            """,
            unsafe_allow_html=True)

        else:

            st.markdown(f"""
            <div style="
            background:#111827;
            border:3px solid #22C55E;
            border-radius:20px;
            padding:25px;
            text-align:center;
            box-shadow:0px 0px 20px #22C55E;
            ">

            <h1 style="color:#22C55E;">
            🐶 DOG
            </h1>

            <h2 style="color:white;">
            {confidence:.2f}% Confidence
            </h2>

            </div>
            """,
            unsafe_allow_html=True)

        st.write("")

        st.markdown("""
        <h3 style="
        text-align:center;
        color:white;
        ">
        Confidence Score
        </h3>
        """,
        unsafe_allow_html=True)

        st.progress(int(confidence))

        st.markdown(
            f"""
            <h2 style="
            text-align:center;
            color:#00FF99;
            ">
            {confidence:.2f}%
            </h2>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # -----------------------------
    # PREDICTION SCORES
    # -----------------------------

    st.markdown("""
    <h2 style="
    text-align:center;
    color:#00E5FF;
    ">
    📊 Prediction Scores
    </h2>
    """,
    unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:

        st.markdown(f"""
        <div style="
        background:#1E293B;
        border:2px solid #38BDF8;
        border-radius:20px;
        padding:20px;
        text-align:center;
        box-shadow:0px 0px 15px #38BDF8;
        ">

        <h3 style="color:white;">
        🐱 Cat
        </h3>

        <h1 style="color:#38BDF8;">
        {prediction[0][0]*100:.2f}%
        </h1>

        </div>
        """,
        unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div style="
        background:#1E293B;
        border:2px solid #22C55E;
        border-radius:20px;
        padding:20px;
        text-align:center;
        box-shadow:0px 0px 15px #22C55E;
        ">

        <h3 style="color:white;">
        🐶 Dog
        </h3>

        <h1 style="color:#22C55E;">
        {prediction[0][1]*100:.2f}%
        </h1>

        </div>
        """,
        unsafe_allow_html=True)

    

st.markdown("---")
# -----------------------------
# MODEL PERFORMANCE
# -----------------------------

st.markdown("""
<h2 style="
text-align:center;
color:#38BDF8;
font-size:40px;
">



</h4>
""", unsafe_allow_html=True)
st.markdown("""
<h2 style="
text-align:center;
color:#38BDF8;
font-size:32px;
">
📈 Accuracy & Loss Graphs
</h2>
""", unsafe_allow_html=True)

with st.expander("📊 Click Here to View Graphs", expanded=False):


    st.image(
        "accuracy.png",
        caption="Training vs Validation Accuracy",
        width=700
    )

    st.image(
        "loss.png",
        caption="Training vs Validation Loss",
        width=700
    )

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# PROJECT SUMMARY
# -----------------------------

st.markdown("""

<div style="
background:#111827;
padding:25px;
border-radius:20px;
border:2px solid #38BDF8;
box-shadow:0px 0px 15px #38BDF8;
">

<h2 style="
text-align:center;
color:#38BDF8;
">

🚀 Project Summary

</h2>

<hr>

<h4 style="color:white;">

🧠 CNN (Convolutional Neural Network)

</h4>

<h4 style="color:white;">

📂 Dataset : Cats vs Dogs

</h4>

<h4 style="color:white;">

⚙️ TensorFlow Deep Learning Framework

</h4>

<h4 style="color:white;">

🌐 Streamlit Web Application

</h4>

<h4 style="color:white;">

📷 Input Size : 128 × 128

</h4>

<h4 style="color:white;">

🤖 AI predicts whether an image belongs to Cat or Dog class.

</h4>

</div>

""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# MODEL INFORMATION
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📂 Classes",
        "2"
    )

with col2:
    st.metric(
        "📷 Image Size",
        "128×128"
    )

with col3:
    st.metric(
        "🧠 Model",
        "CNN"
    )

st.markdown("---")

# -----------------------------
# FOOTER
# -----------------------------

st.markdown("""

<div style="
background:linear-gradient(to right,#0F172A,#1E293B,#0F172A);
padding:25px;
border-radius:20px;
text-align:center;
border:2px solid #22C55E;
">

<h2 style="
color:#22C55E;
">

🐱🐶 Cat vs Dog AI Classifier

</h2>

<h3 style="
color:white;
">

Built using TensorFlow + Streamlit

</h3>

<p style="
color:#CBD5E1;
font-size:18px;
">

Mini Project on Computer Vision using Convolutional Neural Networks (CNN)

</p>

<p style="
color:#38BDF8;
font-size:20px;
">

❤️ Developed by AI & Data Science Student

</p>

</div>

""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# END OF APP
# -----------------------------