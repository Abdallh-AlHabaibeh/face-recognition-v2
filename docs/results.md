# Results

## Controlled Evaluation

The controlled evaluation used single face test images containing both known identities and unknown identities.

| Metric | Value |
|---|---:|
| Total Test Images | 44 |
| Valid Test Images | 43 |
| Skipped Images | 1 |
| Known Test Images | 27 |
| Unknown Test Images | 16 |
| Correct Known Recognitions | 27 |
| Correct Unknown Rejections | 16 |
| False Positives | 0 |
| False Negatives | 0 |
| Wrong Identities | 0 |
| Known Accuracy | 100.00% |
| Unknown Rejection Rate | 100.00% |
| Overall Accuracy | 100.00% |

Detailed results are available in:

- `docs/evaluation_results.csv`

---

## Stress Test

The stress test evaluated the recognition system using group images containing multiple known and unknown individuals. This provides a more realistic assessment of system performance under challenging conditions.

| Metric | Value |
|---|---:|
| Stress Test Images | 24 |
| Detected Faces | 183 |
| Expected Known Faces | 29 |
| Correctly Recognized Known Faces | 27 |
| Missed Known Faces | 2 |
| Unknown-Classified Faces | 156 |
| False Positive Identities | 0 |
| Known-Face Stress Accuracy | 93.10% |

Detailed results are available in:

- `docs/stress_test_results.csv`
- `docs/stress_test_face_results.csv`