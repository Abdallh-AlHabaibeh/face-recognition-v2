import os
import pickle
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import EMBEDDINGS_DIR, MODEL_NAME, MATCH_THRESHOLD, CAMERA_INDEX
from logger import log_recognition


EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")
LOG_INTERVAL_SECONDS = 30


def load_model():
    app = FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=-1, det_size=(320, 320))
    return app


def load_face_database():
    with open(EMBEDDINGS_FILE, "rb") as file:
        return pickle.load(file)


def cosine_distance(embedding1, embedding2):
    embedding1 = embedding1 / np.linalg.norm(embedding1)
    embedding2 = embedding2 / np.linalg.norm(embedding2)
    similarity = np.dot(embedding1, embedding2)
    return 1 - similarity


def find_best_match(live_embedding, face_database):
    best_name = "Unknown"
    best_distance = float("inf")

    for person_name, stored_embeddings in face_database.items():
        for stored_embedding in stored_embeddings:
            distance = cosine_distance(live_embedding, stored_embedding)

            if distance < best_distance:
                best_distance = distance
                best_name = person_name

    if best_distance <= MATCH_THRESHOLD:
        return best_name, best_distance

    return "Unknown", best_distance


def should_log(name, last_logged_times):
    current_time = time.time()

    if name not in last_logged_times:
        last_logged_times[name] = current_time
        return True

    if current_time - last_logged_times[name] >= LOG_INTERVAL_SECONDS:
        last_logged_times[name] = current_time
        return True

    return False


def draw_face_result(frame, face, name, distance):
    x1, y1, x2, y2 = face.bbox.astype(int)
    color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
    label = f"{name} | distance: {distance:.2f}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        label,
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )


def recognize_webcam():
    app = load_model()
    face_database = load_face_database()

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Error: could not open webcam.")
        return

    frame_count = 0
    process_every_n_frames = 5
    last_results = []
    last_logged_times = {}

    print("Webcam started. Press 'q' to quit.")
    print(f"Processing every {process_every_n_frames} frames to reduce lag.")
    print(f"Logging each identity at most once every {LOG_INTERVAL_SECONDS} seconds.")

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error: could not read frame.")
            break

        frame_count += 1

        if frame_count % process_every_n_frames == 0:
            faces = app.get(frame)
            last_results = []

            for face in faces:
                name, distance = find_best_match(face.embedding, face_database)
                status = "Recognized" if name != "Unknown" else "Unknown"

                if should_log(name, last_logged_times):
                    log_recognition(name, distance, status)

                last_results.append((face, name, distance))

        for face, name, distance in last_results:
            draw_face_result(frame, face, name, distance)

        cv2.imshow("Face Recognition V2", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_webcam()