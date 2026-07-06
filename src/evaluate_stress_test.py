import os
import pickle
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

from config import EMBEDDINGS_DIR, MODEL_NAME, MATCH_THRESHOLD


STRESS_TEST_DIR = "data/stress_test"
GROUND_TRUTH_FILE = "data/stress_test_ground_truth.csv"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")

IMAGE_RESULTS_FILE = "docs/stress_test_results.csv"
FACE_RESULTS_FILE = "docs/stress_test_face_results.csv"


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


def parse_expected_identities(value):
    if pd.isna(value) or str(value).strip() == "":
        return []

    return [
        identity.strip()
        for identity in str(value).split(";")
        if identity.strip()
    ]


def load_ground_truth():
    ground_truth_df = pd.read_csv(GROUND_TRUTH_FILE)
    ground_truth = {}

    for _, row in ground_truth_df.iterrows():
        image_name = row["image"]
        expected_identities = parse_expected_identities(
            row["expected_known_identities"]
        )

        ground_truth[image_name] = {
            "expected_known_count": int(row["expected_known_count"]),
            "expected_known_identities": expected_identities
        }

    return ground_truth


def evaluate_stress_test():
    os.makedirs("docs", exist_ok=True)

    app = load_model()
    face_database = load_face_database()
    ground_truth = load_ground_truth()

    face_level_results = []
    image_level_results = []

    for image_name in os.listdir(STRESS_TEST_DIR):
        image_path = os.path.join(STRESS_TEST_DIR, image_name)

        if not os.path.isfile(image_path):
            continue

        image = cv2.imread(image_path)

        if image is None:
            image_level_results.append({
                "image": image_name,
                "expected_known_count": ground_truth.get(
                    image_name, {}
                ).get("expected_known_count", 0),
                "detected_faces": 0,
                "recognized_known_count": 0,
                "recognized_identities": "",
                "missed_known_identities": "",
                "false_positive_identities": "",
                "unknown_count": 0,
                "result_note": "Unreadable image"
            })
            continue

        faces = app.get(image)

        recognized_identities = []
        unknown_count = 0

        for index, face in enumerate(faces, start=1):
            x1, y1, x2, y2 = face.bbox.astype(int)

            h, w = image.shape[:2]

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            bbox_width = max(0, x2 - x1)
            bbox_height = max(0, y2 - y1)

            predicted_identity, distance = find_best_match(
                face.embedding,
                face_database
            )

            if predicted_identity == "Unknown":
                unknown_count += 1
            else:
                recognized_identities.append(predicted_identity)

            face_level_results.append({
                "image": image_name,
                "face_index": index,
                "predicted": predicted_identity,
                "distance": round(distance, 4),
                "bbox_width": bbox_width,
                "bbox_height": bbox_height
            })

        expected_data = ground_truth.get(image_name, {
            "expected_known_count": 0,
            "expected_known_identities": []
        })

        expected_identities = expected_data["expected_known_identities"]

        recognized_unique = list(dict.fromkeys(recognized_identities))

        correctly_recognized = [
            identity
            for identity in expected_identities
            if identity in recognized_unique
        ]

        missed_identities = [
            identity
            for identity in expected_identities
            if identity not in recognized_unique
        ]

        false_positive_identities = [
            identity
            for identity in recognized_unique
            if identity not in expected_identities
        ]

        image_level_results.append({
            "image": image_name,
            "expected_known_count": expected_data["expected_known_count"],
            "detected_faces": len(faces),
            "recognized_known_count": len(correctly_recognized),
            "recognized_identities": ";".join(recognized_unique),
            "missed_known_identities": ";".join(missed_identities),
            "false_positive_identities": ";".join(false_positive_identities),
            "unknown_count": unknown_count,
            "result_note": "Processed"
        })

    face_df = pd.DataFrame(face_level_results)
    image_df = pd.DataFrame(image_level_results)

    face_df.to_csv(FACE_RESULTS_FILE, index=False)
    image_df.to_csv(IMAGE_RESULTS_FILE, index=False)

    total_expected_known = image_df["expected_known_count"].sum()
    total_recognized_known = image_df["recognized_known_count"].sum()

    total_false_positives = image_df["false_positive_identities"].apply(
        lambda value: 0
        if pd.isna(value) or value == ""
        else len(value.split(";"))
    ).sum()

    total_detected_faces = image_df["detected_faces"].sum()
    total_unknown_faces = image_df["unknown_count"].sum()

    known_stress_accuracy = (
        total_recognized_known / total_expected_known * 100
        if total_expected_known > 0
        else 0
    )

    print("\nStress Test Summary")
    print("-------------------")
    print(f"Stress test images: {len(image_df)}")
    print(f"Detected faces: {total_detected_faces}")
    print(f"Expected known faces: {total_expected_known}")
    print(f"Correctly recognized known faces: {total_recognized_known}")
    print(f"Missed known faces: {total_expected_known - total_recognized_known}")
    print(f"Unknown-classified faces: {total_unknown_faces}")
    print(f"False positive identities: {total_false_positives}")
    print(f"Known-face stress accuracy: {known_stress_accuracy:.2f}%")
    print(f"\nImage-level results saved to: {IMAGE_RESULTS_FILE}")
    print(f"Face-level results saved to: {FACE_RESULTS_FILE}")


if __name__ == "__main__":
    evaluate_stress_test()