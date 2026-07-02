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

