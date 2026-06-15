import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ======================================
# LOAD MODEL
# ======================================

MODEL_PATH = "models/cnn_model.h5"

model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (128, 128)

CLASS_NAMES = ["🐱 Cat", "🐶 Dog"]


# ======================================
# PREDICTION FUNCTION
# ======================================

def predict(img):

    if img is None:
        return (
            "",
            "",
            {"Cat": 0, "Dog": 0},
            "outputs/accuracy.png",
            "outputs/loss.png"
        )

    img = img.convert("RGB")
    img = img.resize((128,128))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)[0]

    cat = float(prediction[0])
    dog = float(prediction[1])

    if cat > dog:
        label = "🐱 CAT"
        confidence = cat * 100
    else:
        label = "🐶 DOG"
        confidence = dog * 100

    scores = {
        "Cat": round(cat * 100, 2),
        "Dog": round(dog * 100, 2)
    }

    return (
        label,
        f"{confidence:.2f}% Confidence",
        scores,
        "outputs/accuracy.png",
        "outputs/loss.png"
    )
# ======================================
# CUSTOM CSS
# ======================================

css = """

body{
background:#020617;
color:white;
}

.gradio-container{
background:linear-gradient(180deg,#020617,#071330,#020617);
color:white !important;
font-family:Segoe UI;
}

ul{
color:white !important;
}

li{
color:white !important;
}

div{
color:white !important;
}

p{
color:white !important;
}

span{
color:white !important;
}
table,
tr,
td,
th {
    color: white !important;
}

td * {
    color: white !important;
}

"""


# ======================================
# BUILD UI
# ======================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    css=css,
    title="Cat vs Dog AI Classifier"
) as demo:

    gr.HTML("""

<div style="text-align:center">

<h1 style="
font-size:45px;
font-weight:bold;
color:white;
text-align:center;
">
🐱🐶 Cat vs Dog AI Classifier
</h1>

<h3 style="color:white;">
🚀 Powered by Deep Learning (CNN)
</h3>

<p style="
color:white;
font-size:18px;
">

Upload an image and let AI identify
whether it is a
<b style="color:white;">
Cat
</b>
or
<b style="color:white;">
Dog
</b>.

</p>

</div>

""")

    with gr.Row():

        with gr.Column(scale=1):

            image = gr.Image(
                type="pil",
                label="📂 Upload Image",
                height=350
            )

        with gr.Column(scale=1):

            prediction = gr.Textbox(
                label="🤖 AI Prediction",
                interactive=False
            )

            confidence = gr.Textbox(
                label="Confidence",
                interactive=False
            )

            scores = gr.Label(
                label="📊 Prediction Scores"
            )

            predict_btn = gr.Button(
                "🔍 Predict",
                variant="primary"
            )
    # ======================================
    # MODEL PERFORMANCE
    # ======================================

    gr.Markdown("---")

    gr.HTML("""

    <div style="
    background:#071330;
    padding:15px;
    border-radius:15px;
    border:2px solid white;
    margin-top:15px;
    ">

    <h2 style="
    text-align:center;
    color:white;
    ">

    📈 Model Performance Graphs

    </h2>

    </div>

    """)

    with gr.Row():

        accuracy_img = gr.Image(
            value="outputs/accuracy.png",
            label="📊 Training vs Validation Accuracy",
            interactive=False,
            height=300
        )

        loss_img = gr.Image(
            value="outputs/loss.png",
            label="📉 Training vs Validation Loss",
            interactive=False,
            height=300
        )

    gr.Markdown("---")
    # ======================================
    # PROJECT SUMMARY
    # ======================================

    gr.HTML("""

    <div style="
    background:#071330;
    border:2px solid #22C55E;
    border-radius:15px;
    padding:20px;
    margin-top:20px;
    ">

    <h2 style="
    text-align:center;
    color:white;
    ">

    🚀 Project Summary

    </h2>

    <ul style="
color:white !important;
font-size:18px;
line-height:2;
">

<li style="color:white;">🧠 Deep Learning Model : CNN</li>

<li style="color:white;">📁 Dataset : Cats vs Dogs</li>

<li style="color:white;">⚙️ Framework : TensorFlow / Keras</li>

<li style="color:white;">🌐 Frontend : Gradio</li>

<li style="color:white;">📷 Input Image Size : 128 × 128</li>

<li style="color:white;">🤖 AI predicts whether the uploaded image is Cat or Dog</li>

</ul>

    </div>

    """)

    # ======================================
    # MODEL INFORMATION
    # ======================================

    gr.HTML("""

<div style="
background:#071330;
border:2px solid #38BDF8;
border-radius:15px;
padding:20px;
margin-top:20px;
">

<h2 style="text-align:center;color:white;">
📊 Model Information
</h2>

<table style="width:100%;font-size:18px;">

<tr>
<td style="color:white;font-weight:bold;">📁 Classes</td>
<td style="color:white;">2 (Cat & Dog)</td>
</tr>

<tr>
<td style="color:white;font-weight:bold;">🖼 Image Size</td>
<td style="color:white;">128 × 128</td>
</tr>

<tr>
<td style="color:white;font-weight:bold;">🧠 Algorithm</td>
<td style="color:white;">Convolutional Neural Network (CNN)</td>
</tr>

<tr>
<td style="color:white;font-weight:bold;">⚡ Backend</td>
<td style="color:white;">TensorFlow</td>
</tr>

<tr>
<td style="color:white;font-weight:bold;">💾 Model Format</td>
<td style="color:white;">.h5</td>
</tr>

</table>

</div>

""")
    # ======================================
    # FOOTER
    # ======================================

    gr.HTML("""

    <div style="
    background:linear-gradient(135deg,#071330,#0B1120,#071330);
    border:2px solid #22C55E;
    border-radius:18px;
    padding:25px;
    margin-top:25px;
    text-align:center;
    box-shadow:0px 0px 20px rgba(34,197,94,0.4);
    ">

    <h1 style="
    color:white;
    font-size:34px;
    margin-bottom:8px;
    ">

    🐱🐶 Cat vs Dog AI Classifier

    </h1>

    <h3 style="
    color:white;
    ">

    Built using TensorFlow + Gradio

    </h3>

    <p style="
    color:white;
    font-size:18px;
    ">

    Mini Project on Computer Vision using
    Convolutional Neural Networks (CNN)

    </p>

    <hr style="
    border:1px solid #334155;
    ">

    <div style="
    display:flex;
    justify-content:center;
    gap:30px;
    flex-wrap:wrap;
    margin-top:15px;
    color:white;
    font-size:18px;
    ">

        <span>🐍 Python</span>

        <span>🧠 TensorFlow</span>

        <span>🎨 Gradio</span>

        <span>🤖 AI</span>

        <span>📷 Computer Vision</span>

    </div>

    <p style="
    margin-top:18px;
    color:white;
    font-size:18px;
    ">

    ❤️ Developed by AI & Data Science Student

    </p>

    </div>

    """)
    # ======================================
    # BUTTON ACTION
    # ======================================

    predict_btn.click(
        fn=predict,
        inputs=image,
        outputs=[
            prediction,
            confidence,
            scores,
            accuracy_img,
            loss_img
        ]
    )
# ======================================
# LAUNCH APPLICATION
# ======================================

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
        )