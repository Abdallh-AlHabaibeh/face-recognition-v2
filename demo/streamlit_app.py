import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
EMBEDDINGS_FILE = ROOT_DIR / "data" / "embeddings" / "face_embeddings.pkl"

sys.path.insert(0, str(SRC_DIR))

from recognize_webcam import find_best_match, load_model


DISPLAY_NAMES = {
    "Achraf_Hakimi": "Achraf Hakimi",
    "Aitana_Bonmati": "Aitana Bonmatí",
    "Alexia_Putellas": "Alexia Putellas",
    "Cristiano_ronaldo": "Cristiano Ronaldo",
    "Haaland": "Erling Haaland",
    "Karim_Benzema": "Karim Benzema",
    "Lionel_messi": "Lionel Messi",
    "Luka_Modric": "Luka Modrić",
    "Mbappe": "Kylian Mbappé",  
    "Neymar": "Neymar Jr.",
    "Salma_Paralluelo": "Salma Paralluelo",
    "Vinicius": "Vinícius Júnior",
    "pep_guardiola": "Pep Guardiola",
    "zlatan": "Zlatan Ibrahimović",
}


st.set_page_config(
    page_title="Face Recognition V2",
    page_icon="👤",
    layout="centered",
)


st.markdown(
    """
    <style>

    html,
    body,
    [data-testid="stAppViewContainer"],
    .stApp {
        background: #0f172a !important;
        color: #e5e7eb !important;
    }

    header[data-testid="stHeader"] {
        display: none !important;
    }

    div[data-testid="stToolbar"] {
        display: none !important;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 18% 10%,
                rgba(59, 130, 246, 0.10),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 22%,
                rgba(99, 102, 241, 0.08),
                transparent 30%
            ),
            linear-gradient(
                180deg,
                #0f172a 0%,
                #111827 45%,
                #0b1220 100%
            ) !important;
    }

    .block-container {
        max-width: 950px;
        padding-top: 2.8rem;
        padding-bottom: 4rem;
    }

    h1,
    h2,
    h3 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    p,
    label {
        color: #cbd5e1 !important;
    }

    .subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        line-height: 1.7;
        margin-bottom: 1.3rem;
    }

    .info-box {
        background: rgba(15, 23, 42, 0.60);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        padding: 18px 20px;
        margin: 1rem 0 1.4rem 0;
    }

    .info-title {
        color: #f8fafc;
        font-weight: 650;
        margin-bottom: 8px;
    }

    .info-text {
        color: #94a3b8;
        line-height: 1.6;
        font-size: 0.94rem;
    }

    .identity-list {
        color: #cbd5e1;
        line-height: 1.7;
        margin-top: 10px;
    }

    .loading-box {
        background: rgba(15, 23, 42, 0.60);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 16px;
        padding: 18px 20px;
        color: #cbd5e1;
        margin: 1rem 0;
    }

    .result-card {
        background: rgba(15, 23, 42, 0.68);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 16px;
        padding: 16px 18px;
        margin-top: 12px;
    }

    .result-known {
        border-left: 3px solid #60a5fa;
    }

    .result-unknown {
        border-left: 3px solid #94a3b8;
    }

    .result-name {
        color: #f8fafc;
        font-size: 1.08rem;
        font-weight: 650;
    }

    .result-detail {
        color: #94a3b8;
        margin-top: 4px;
        font-size: 0.92rem;
    }

    div[data-testid="stFileUploader"] {
        background: transparent !important;
        border-radius: 16px;
    }

    div[data-testid="stFileUploader"] section {
        background: rgba(15, 23, 42, 0.72) !important;
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
        border-radius: 14px !important;
    }

    div[data-testid="stFileUploader"] section * {
        color: #cbd5e1 !important;
    }

    div[data-testid="stFileUploader"] button {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 10px !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border-color: rgba(96, 165, 250, 0.35) !important;
    }

    div[data-testid="stCameraInput"] {
        background: rgba(15, 23, 42, 0.72) !important;
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
        border-radius: 14px !important;
        overflow: hidden;
    }

    div[data-testid="stCameraInput"] button {
        background: rgba(30, 41, 59, 0.95) !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
    }

    div[data-testid="stCameraInput"] button:hover {
        background: rgba(51, 65, 85, 0.95) !important;
    }

    button[data-baseweb="tab"] {
        color: #cbd5e1 !important;
    }

    div[data-testid="stSpinner"] {
        color: #cbd5e1 !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_model()


@st.cache_resource(show_spinner=False)
def get_face_database():
    with EMBEDDINGS_FILE.open("rb") as file:
        return pickle.load(file)


def decode_image(uploaded_file):
    file_bytes = np.asarray(
        bytearray(uploaded_file.getvalue()),
        dtype=np.uint8,
    )

    return cv2.imdecode(
        file_bytes,
        cv2.IMREAD_COLOR,
    )


def get_display_name(name):
    if name == "Unknown":
        return "Unknown"

    return DISPLAY_NAMES.get(
        name,
        name.replace("_", " "),
    )


def recognize_faces(image, model, face_database):
    output = image.copy()
    faces = model.get(image)
    results = []

    for face in faces:
        name, distance = find_best_match(
            face.embedding,
            face_database,
        )

        x1, y1, x2, y2 = face.bbox.astype(int)

        height, width = output.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        display_name = get_display_name(name)

        color = (
            (0, 0, 255)
            if name == "Unknown"
            else (0, 255, 0)
        )

        label = display_name

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            2,
        )

        cv2.putText(
            output,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )

        results.append(
            {
                "name": name,
                "display_name": display_name,
                "distance": distance,
            }
        )

    return output, results


def show_results(image, model, face_database):
    with st.spinner(
        "Processing image and comparing detected faces..."
    ):
        output, results = recognize_faces(
            image,
            model,
            face_database,
        )

    output_rgb = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB,
    )

    st.image(
        output_rgb,
        width=620,
    )

    if not results:
        st.warning(
            "No faces were detected in this image."
        )
        return

    st.subheader("Recognition Results")

    for index, result in enumerate(results, start=1):
        display_name = result["display_name"]
        distance = result["distance"]

        if result["name"] == "Unknown":
            card_class = "result-card result-unknown"
            status = "No enrolled identity matched this face."
        else:
            card_class = "result-card result-known"
            status = "Matched with an enrolled identity."

        st.html(
            f"""
            <div class="{card_class}">
                <div class="result-name">
                    Face {index}: {display_name}
                </div>
                <div class="result-detail">
                    {status}
                </div>
                <div class="result-detail">
                    Cosine distance: {distance:.3f}
                </div>
            </div>
            """
        )


st.title("Face Recognition V2")

st.html(
    """
    <div class="subtitle">
        Detect and recognize faces from uploaded images or photos captured directly in your browser.
    </div>
    """
)


loading_placeholder = st.empty()

with loading_placeholder.container():
    st.html(
        """
        <div class="loading-box">
            Loading recognition model and enrolled identities...
        </div>
        """
    )

model = get_model()
face_database = get_face_database()

loading_placeholder.empty()


known_identities = sorted(
    get_display_name(name)
    for name in face_database.keys()
)

identity_text = " • ".join(known_identities)


st.html(
    f"""
    <div class="info-box">
        <div class="info-title">
            Recognition database
        </div>
        <div class="info-text">
            This demo recognizes only the identities enrolled in the
            database. Any other detected face is classified as
            <strong>Unknown</strong>.
        </div>
        <div class="identity-list">
            {identity_text}
        </div>
    </div>
    """
)


upload_tab, camera_tab = st.tabs(
    ["Upload Image", "Take a Photo"]
)


with upload_tab:
    st.html(
        """
        <div class="info-box">
            <div class="info-title">
                Upload Image
            </div>
            <div class="info-text">
                Select a JPG or PNG image from your device.
                Processing starts automatically after the image is uploaded.
                The app detects every visible face and compares each one
                against the enrolled identities.
            </div>
        </div>
        """
    )

    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded_image is not None:
        image = decode_image(uploaded_image)

        if image is None:
            st.error(
                "The uploaded image could not be read."
            )
        else:
            show_results(
                image,
                model,
                face_database,
            )


with camera_tab:
    st.html(
        """
        <div class="info-box">
            <div class="info-title">
                Take a Photo
            </div>
            <div class="info-text">
                Allow browser camera access and take a single photo.
                After the photo is captured, the model automatically
                detects and recognizes the visible faces.
            </div>
        </div>
        """
    )

    camera_image = st.camera_input(
        "Take a photo",
        label_visibility="collapsed",
    )

    if camera_image is not None:
        image = decode_image(camera_image)

        if image is None:
            st.error(
                "The captured image could not be read."
            )
        else:
            show_results(
                image,
                model,
                face_database,
            )