# Face Recognition V2

A real time face recognition system built with Python, OpenCV, and InsightFace.

The program creates facial embeddings from a local image dataset and uses them to recognize known individuals from a webcam feed. Recognition events are automatically logged to a CSV file.

---

## Features

- Generate facial embeddings from a local image dataset.
- Perform real time webcam face recognition.
- Classify faces as **Recognized** or **Unknown**.
- Log recognition events to a CSV file.

---

## Technologies

- Python
- OpenCV
- InsightFace
- ONNX Runtime
- NumPy

---

## Project Structure

```text
face-recognition-v2/
│
├── src/
├── data/
├── logs/
├── README.md
└── requirements.txt
```

---

## Usage

Generate the face embeddings:

```bash
python src/encode_faces.py
```

Start the recognition system:

```bash
python src/recognize_webcam.py
```

Recognition events are saved to:

```text
logs/recognition_log.csv
```

---

## Dataset

The project expects one folder per person inside `data/raw_faces/`, with multiple images for each identity.

---

## Evaluation

The recognition system was evaluated using two dedicated evaluation scripts:

- `src/evaluate_model.py` for controlled testing.
- `src/evaluate_stress_test.py` for look alike and group image stress testing.

The evaluation uses separate controlled and stress test datasets located in the `data/` directory.

### Controlled Evaluation Results

| Metric | Result |
|---|---:|
| Known Accuracy | 100.00% |
| Unknown Rejection Rate | 100.00% |
| Overall Accuracy | 100.00% |

---

## Stress Testing

To evaluate the system under more realistic conditions, additional stress tests were performed using group images containing multiple known and unknown individuals.

Each stress-test image is paired with a manually created ground-truth file (`stress_test_ground_truth.csv`) containing the expected known identities. This allows the system to compare predictions against the expected results and calculate recognition accuracy.

### Stress Test Results

| Metric | Result |
|---|---:|
| Stress Test Images | 24 |
| Detected Faces | 183 |
| Expected Known Faces | 29 |
| Correctly Recognized Known Faces | 25 |
| False Positive Identities | 0 |
| Known-Face Stress Accuracy | 86.21% |

Detailed evaluation outputs are available in the `docs/` directory, including image level and face level evaluation reports.

---