import os
import pickle
import cv2
import numpy as np
import pandas as pd
from insightface.app import FaceAnalysis

from config import TEST_IMAGES_DIR, EMBEDDINGS_DIR, MODEL_NAME, MATCH_THRESHOLD


EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")
RESULTS_FILE = "docs/evaluation_results.csv"


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


def get_expected_identity(filename):
    name_without_extension = os.path.splitext(filename)[0]

    parts = name_without_extension.split("_")

    expected_identity = "_".join(parts[:-1])

    if expected_identity.lower() == "unknown":
        return "Unknown"

    return expected_identity


def evaluate_model():
    app = load_model()
    face_database = load_face_database()

    results = []

    for image_name in os.listdir(TEST_IMAGES_DIR):
        image_path = os.path.join(TEST_IMAGES_DIR, image_name)

        if not os.path.isfile(image_path):
            continue

        expected_identity = get_expected_identity(image_name)

        image = cv2.imread(image_path)

        if image is None:
            results.append({
                "image": image_name,
                "expected": expected_identity,
                "predicted": "Unreadable",
                "distance": None,
                "result": "Skipped"
            })
            continue

        faces = app.get(image)

        if len(faces) != 1:
            results.append({
                "image": image_name,
                "expected": expected_identity,
                "predicted": f"{len(faces)} faces detected",
                "distance": None,
                "result": "Skipped"
            })
            continue

        predicted_identity, distance = find_best_match(
            faces[0].embedding,
            face_database
        )

        if expected_identity == "Unknown":
            if predicted_identity == "Unknown":
                result = "Correct Unknown"
            else:
                result = "False Positive"
        else:
            if predicted_identity == expected_identity:
                result = "Correct Recognition"
            elif predicted_identity == "Unknown":
                result = "False Negative"
            else:
                result = "Wrong Identity"

        results.append({
            "image": image_name,
            "expected": expected_identity,
            "predicted": predicted_identity,
            "distance": round(distance, 4),
            "result": result
        })

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_FILE, index=False)

    total_tests = len(df)
    skipped = len(df[df["result"] == "Skipped"])

    valid_df = df[df["result"] != "Skipped"]

    correct_results = valid_df[
        valid_df["result"].isin(["Correct Recognition", "Correct Unknown"])
    ]

    known_df = valid_df[valid_df["expected"] != "Unknown"]
    unknown_df = valid_df[valid_df["expected"] == "Unknown"]

    correct_known = len(
        known_df[known_df["result"] == "Correct Recognition"]
    )

    correct_unknown = len(
        unknown_df[unknown_df["result"] == "Correct Unknown"]
    )

    false_positives = len(valid_df[valid_df["result"] == "False Positive"])
    false_negatives = len(valid_df[valid_df["result"] == "False Negative"])
    wrong_identities = len(valid_df[valid_df["result"] == "Wrong Identity"])

    overall_accuracy = len(correct_results) / len(valid_df) * 100

    known_accuracy = correct_known / len(known_df) * 100 if len(known_df) > 0 else 0
    unknown_rejection_rate = (
        correct_unknown / len(unknown_df) * 100 if len(unknown_df) > 0 else 0
    )

    print("\nEvaluation Summary")
    print("------------------")
    print(f"Total test images: {total_tests}")
    print(f"Valid test images: {len(valid_df)}")
    print(f"Skipped images: {skipped}")
    print(f"Known test images: {len(known_df)}")
    print(f"Unknown test images: {len(unknown_df)}")
    print(f"Correct known recognitions: {correct_known}")
    print(f"Correct unknown rejections: {correct_unknown}")
    print(f"False positives: {false_positives}")
    print(f"False negatives: {false_negatives}")
    print(f"Wrong identities: {wrong_identities}")
    print(f"Known accuracy: {known_accuracy:.2f}%")
    print(f"Unknown rejection rate: {unknown_rejection_rate:.2f}%")
    print(f"Overall accuracy: {overall_accuracy:.2f}%")
    print(f"\nDetailed results saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    evaluate_model()