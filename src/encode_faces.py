import os
import pickle
import cv2
from insightface.app import FaceAnalysis

from config import RAW_FACES_DIR, EMBEDDINGS_DIR, MODEL_NAME


OUTPUT_FILE = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")


def load_model():
    app = FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=-1, det_size=(320, 320))  # CPU mode
    return app


def encode_faces():
    os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

    app = load_model()
    face_database = {}

    for person_name in os.listdir(RAW_FACES_DIR):
        person_folder = os.path.join(RAW_FACES_DIR, person_name)

        if not os.path.isdir(person_folder):
            continue

        face_database[person_name] = []

        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)

            image = cv2.imread(image_path)

            if image is None:
                print(f"Skipped unreadable image: {image_path}")
                continue

            faces = app.get(image)

            if len(faces) != 1:
                print(f"Skipped {image_path}: detected {len(faces)} faces")
                continue

            embedding = faces[0].embedding
            face_database[person_name].append(embedding)

            print(f"Encoded: {person_name} / {image_name}")

        if len(face_database[person_name]) == 0:
            print(f"Warning: no valid embeddings found for {person_name}")

    with open(OUTPUT_FILE, "wb") as file:
        pickle.dump(face_database, file)

    print(f"\nSaved embeddings to: {OUTPUT_FILE}")


if __name__ == "__main__":
    encode_faces()