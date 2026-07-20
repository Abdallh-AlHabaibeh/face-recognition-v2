# Methodology

## Overview

The face recognition system follows a feature-based recognition pipeline using InsightFace embeddings and cosine distance matching. The workflow consists of four main stages:

1. Face database creation.
2. Face detection and recognition.
3. Controlled evaluation.
4. Stress testing.

---

## Face Database Creation

The face database is created from images stored in the `data/raw_faces/` directory. Each person's images are placed inside a separate folder named after the corresponding identity.

Example:

```text
data/raw_faces/
├── Cristiano_ronaldo/
├── Lionel_messi/
├── Mbappe/
└── ...
```

For each image, the system:

1. Detects the face using InsightFace.
2. Generates a facial embedding.
3. Stores the embedding with its associated identity.

All embeddings are saved in a serialized database:

```text
data/embeddings/face_embeddings.pkl
```

This preprocessing step allows the recognition system to avoid recomputing embeddings every time the application starts.

---

## Face Recognition

During recognition, frames are continuously captured from the webcam.

For every detected face, the system:

1. Detects the face.
2. Generates a facial embedding.
3. Compares the embedding against every stored embedding in the database using cosine distance.
4. Selects the closest matching identity.
5. Compares the distance against the recognition threshold.
6. Returns either the predicted identity or **Unknown**.

Recognition events are recorded in:

```text
logs/recognition_log.csv
```

---

## Similarity Measurement

Identity matching is performed using cosine distance between facial embeddings.

The stored embedding with the smallest cosine distance is selected as the best candidate.

To reduce false recognitions, the prediction is only accepted when the distance is below the predefined matching threshold.

Final threshold:

```text
MATCH_THRESHOLD = 0.63
```

Faces exceeding this threshold are classified as **Unknown**.

---

## Controlled Evaluation

The recognition system was evaluated using a controlled dataset containing individual images of both known and unknown identities.

The evaluation process:

1. Load the embedding database.
2. Load each test image.
3. Detect the face.
4. Generate an embedding.
5. Predict the identity.
6. Compare the prediction with the expected label.
7. Record the result.

The expected identity is automatically extracted from the image filename.

Example:

```text
Lionel_messi_1.jpg
```

Expected identity:

```text
Lionel_messi
```

The evaluation reports:

- Known recognition accuracy
- Unknown rejection rate
- Overall accuracy
- False positives
- False negatives

Detailed results are saved in:

```text
docs/evaluation_results.csv
```

---

## Stress Testing

A second evaluation stage measures performance under more challenging conditions.

The stress-test dataset contains:

- Group images
- Multiple faces
- Known identities
- Unknown individuals
- Look-alike images

Unlike the controlled evaluation, expected identities cannot be inferred from filenames because multiple people may appear in a single image.

Instead, each image is paired with a manually created ground-truth file:

```text
data/stress_test_ground_truth.csv
```

The ground-truth file stores:

- Image name
- Expected known-face count
- Expected known identities

The stress-test script compares predictions against these expected identities to calculate recognition performance.

---

## Performance Metrics

Two evaluation reports are generated during stress testing.

### Image-Level Report

```text
docs/stress_test_results.csv
```

Contains one record for each evaluated image, including:

- Expected known faces
- Detected faces
- Correct recognitions
- Missed identities
- False positives
- Unknown classifications

### Face-Level Report

```text
docs/stress_test_face_results.csv
```

Contains one record for every detected face, including:

- Predicted identity
- Cosine distance
- Bounding box dimensions

---

## Evaluation Summary

The final implementation achieved:

### Controlled Evaluation

- Known Accuracy: **100.00%**
- Unknown Rejection Rate: **100.00%**
- Overall Accuracy: **100.00%**

### Stress Testing

- Stress Test Images: **24**
- Detected Faces: **183**
- Expected Known Faces: **29**
- Correctly Recognized Known Faces: **27**
- Missed Known Faces: **2**
- False Positive Identities: **0**
- Known-Face Stress Accuracy: **93.10%**

These evaluations demonstrate that the system performs reliably under controlled conditions while maintaining strong performance in more realistic multi-person scenarios.

---

## Limitations

The current implementation has several limitations:

- Recognition performance depends on image quality and face size.
- Large pose variations and occlusions may reduce recognition accuracy.
- The evaluation datasets are relatively small.
- The stress-test ground truth is manually annotated.
- The system currently operates on a local embedding database and is not designed for large-scale deployments.

Despite these limitations, the project demonstrates a complete end-to-end face recognition pipeline, including dataset preparation, real-time recognition, quantitative evaluation, and performance analysis.