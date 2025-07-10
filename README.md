# Animal Image Classification with Custom CNNs

This project explores the performance of several custom Convolutional Neural Network (CNN) architectures on a multi-class animal image classification task. The models are built from scratch using PyTorch and compared across various performance metrics.

## Project Structure

- **Custom CNN Architectures**: Three progressively complex CNN models (ModelV0, ModelV1, ModelV2) are implemented and trained.
- **Baseline Comparison**: A `RandomGuessModel` is introduced to serve as a non-learned baseline.
- **Training & Evaluation**: Training loss, validation loss, accuracy, F1-score, and confusion matrices are used to evaluate model performance.
- **Visualization**: The notebook provides helpful visualizations including loss/accuracy curves, sample predictions, and confusion matrices.

## Model Architectures

| Model       | Conv Blocks | Dropout | Fully Connected Layers  | Description                               |
|-------------|-------------|---------|-------------------------|-------------------------------------------|
| ModelV0     | 2           | 0       | 1                       | Simple baseline CNN                       |
| ModelV1     | 3           | 1       | 1                       | Improved CNN with dropout                 |
| ModelV2     | 4           | 2       | 2                       | Deep CNN with multiple layers and dropout |
| RandomGuess | 0           | 0       | 0                       | Predicts classes randomly for comparison  |

## Results

All models were evaluated on a hold-out test set using accuracy, confusion matrix, and F1-macro score.

### Highlights:
- ModelV2 achieved the highest F1-macro score and best generalization.
- Class 'cat' was consistently the hardest to classify across all models.
- Random guessing served as a sanity check and performed significantly worse.

## Visualizations

- Training/Validation Loss & Accuracy Curves
- Correct vs Incorrect Predictions
- Per-class Confusion Matrices
- F1-Macro Scores for Each Model

## Tools Used

- Python 3.12 (Jupyter Notebook)
- PyTorch
- torchvision
- scikit-learn
- numpy
- pandas
- matplotlib
- seaborn


## Summary

This project demonstrates how even simple custom-built CNNs can achieve solid performance on image classification tasks when designed thoughtfully. The step-by-step comparison of models provides insight into the impact of architectural choices like depth, dropout, and layer structure.