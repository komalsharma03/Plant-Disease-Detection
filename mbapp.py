import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
st.write(__file__)
# PAGE CONFIG
st.set_page_config(
    page_title=" Plant Disease Detector",
   layout="wide"
)

# CSS
st.markdown("""
<style>

.stApp{
background:linear-gradient(
135deg,
#d7f9d7,
#ffffff
);
}

.title{
font-size:50px;
font-weight:800;
text-align:center;
color:#0a7c34;
}

.subtitle{
text-align:center;
color:#555;
font-size:18px;
margin-bottom:30px;
}

.card{
background:white;
padding:25px;

border-radius:20px;

box-shadow:
0 6px 20px
rgba(0,0,0,.12);

}

.result{
font-size:32px;

font-weight:700;

color:#0f6b2f;

}

.conf{
font-size:22px;

color:#444;
}

.footer{

text-align:center;

padding-top:30px;

color:gray;

}

</style>
""", unsafe_allow_html=True)

# HEADER


st.markdown(
"""
<div class='title'>
 Plant Disease Detection
</div>

<div class='subtitle'>
Deep Learning + EfficientNetB0 + Streamlit
</div>
""",
unsafe_allow_html=True
)


# SIDEBAR


with st.sidebar:

    st.title("Project")

    st.write("""
Detect diseases from plant leaves.

Upload Image  
AI Prediction  
Confidence Score  
Instant Results
""")

# PATHS

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "efficientnetb0_plant.pth"
)

CLASS_PATH = os.path.join(
    BASE_DIR,
    "plant_classes.npy"
)

# LOAD CLASSES

if not os.path.exists(CLASS_PATH):

    st.error(
        "plant_classes.npy missing"
    )

    st.stop()

class_names = np.load(
    CLASS_PATH,
    allow_pickle=True
)

NUM_CLASSES = len(
    class_names
)

# MODEL
@st.cache_resource
def load_model():

    model = models.efficientnet_b0(
        weights=None
    )

    model.classifier[1] = nn.Linear(
        1280,
        NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    if "state_dict" in checkpoint:

        checkpoint = checkpoint[
            "state_dict"
        ]

    model.load_state_dict(
        checkpoint,
        strict=True
    )

    model.eval()

    return model


with st.spinner(
    "Loading AI Model..."
):

    model = load_model()

# TRANSFORM

transform = transforms.Compose([

    transforms.Resize(
        (
            224,
            224
        )
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        [0.485,0.456,0.406],

        [0.229,0.224,0.225]

    )

])

# PREDICT

def predict(img):

    image = transform(
        img
    )

    image = image.unsqueeze(
        0
    )

    with torch.no_grad():

        output = model(
            image
        )

        prob = torch.softmax(
            output,
            dim=1
        )

        idx = torch.argmax(
            prob,
            dim=1
        ).item()

    return (

        class_names[idx],

        prob[0][idx].item()

    )

# MAIN

left,right = st.columns(
    [1,1]
)

with left:

    uploaded = st.file_uploader(

        "Upload Plant Image",

        type=[

            "jpg",

            "jpeg",

            "png"

        ]

    )

    if uploaded:

        img = Image.open(
            uploaded
        ).convert(
            "RGB"
        )

        st.image(

            img,

            use_container_width=True

        )

with right:

    if uploaded:

        st.write("")

        st.write("")

        if st.button(

            "Detect Plant Disease",

            use_container_width=True

        ):

            with st.spinner(

                "Analyzing..."

            ):

                label,conf = predict(
                    img
                )

            st.markdown(

f"""
<div class='card'>

<div class='result'>
 {label}
</div>

<br>

<div class='conf'>
Confidence:
{conf:.2%}
</div>

</div>
""",

unsafe_allow_html=True

)

            st.progress(
                float(conf)
            )

            if conf>.90:

                st.success(
                    "Healthy Prediction Confidence"
                )

            elif conf>.70:

                st.info(
                    "Moderate Confidence"
                )

            else:

                st.warning(
                    "Low Confidence"
                )

# FOOTER

st.markdown(
"""
<div class='footer'>
Made with using Streamlit + PyTorch
</div>
""",
unsafe_allow_html=True
)