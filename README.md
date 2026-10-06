# Cat vs Dog Classifier

An end-to-end deep learning project that classifies images as a cat or a dog. It covers automated data ingestion, model training and evaluation, experiment tracking with MLflow, a FastAPI prediction service, and a web interface for uploading images and viewing predictions.

Two models were built and compared on a dataset of 25,000 images:

- A custom CNN trained from scratch
- A ResNet-18 model, which reached 96% overall accuracy


## What This Project Does

- Downloads the dataset from Google Drive and extracts it automatically.
- Trains a custom CNN from scratch in PyTorch, using convolutional blocks, batch normalization, and dropout.
- Trains a ResNet-18 model and compares it against the custom CNN.
- Evaluates both models using accuracy, precision, and recall.
- Organizes the code into modular pipeline stages: data ingestion, training, and evaluation.
- Tracks experiments with a remote MLflow server deployed using Docker and Docker Compose.
- Serves predictions through a FastAPI backend with CORS support.
- Provides a web interface with drag-and-drop upload, image preview, and real-time predictions with confidence scores.
- Includes logging, YAML-based configuration, and error handling.

## Results

Both models were trained on a dataset of 25,000 cat and dog images.

| Metric | Custom CNN | ResNet-18 |
|---|---|---|
| Training accuracy | 80% | 98% |
| Testing accuracy | 78% | 95% |
| Overall accuracy | 78% | 96% |
| Precision | 80% | 96% |
| Recall | 78% | 96% |

Key observations:

- ResNet-18 improves overall accuracy by 18 percentage points over the custom CNN.
- The gap between training and testing accuracy is small for both models (2 points for the CNN, 3 points for ResNet-18), so neither is heavily overfitting.
- The custom CNN is limited mainly by its shallow architecture and short training (5 epochs), not by overfitting.
- ResNet-18 has balanced precision and recall (both 96%), so it is not biased toward either class.

## Project Structure

```
cat-vs-dog-classifier/
├── api/                      # FastAPI backend
│   └── app.py                # API with /predict endpoint
├── artifact/                 # Generated data and models
│   ├── data_ingestion/       # Downloaded and extracted dataset
│   └── model/                # Trained model.pth file
├── src/Classifier/           # Core ML code
│   ├── components/           # Data ingestion, model, training, evaluation
│   ├── config/               # Configuration manager
│   ├── entity/               # Configuration dataclasses
│   ├── pipeline/             # Pipeline stage classes
│   └── utils/                # YAML, file, and logging helpers
├── templates/                # Frontend
│   └── index.html            # Prediction UI
├── notebooks/                # Research and experimentation
│   └── research.ipynb
├── config/
│   └── config.yaml           # Pipeline configuration
├── logging/                  # Log files
├── requirements.txt          # Python dependencies
└── main.py                   # Runs the full pipeline
```

## Getting Started

### Prerequisites

- Python 3.12
- uv (or pip with `requirements.txt`)
- Internet connection to download the dataset
- Docker and Docker Compose (only needed for the MLflow server)

### Installation

```bash
git clone <repository-url>
cd cat-vs-dog-classifier

uv sync
source .venv/bin/activate
```

Using pip instead:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training the Model

Run the complete pipeline (data ingestion, training, evaluation):

```bash
python main.py
```
or 
```bash
dvc repro
```

This downloads and extracts the dataset to `artifact/data_ingestion/`, trains the model, and saves the weights to `artifact/model/model.pth`. Paths and settings are controlled from `config/config.yaml`.

## Running the API

```bash
uv run uvicorn api.app:app --reload
```

The API runs at `http://127.0.0.1:8000`. Interactive documentation is available at `http://127.0.0.1:8000/docs`.

## Running the Frontend

With the API running, serve the frontend in a second terminal:

```bash
cd templates
python -m http.server 3000
```

Open `http://localhost:3000` in your browser, upload an image, and view the predicted class with its confidence score. The frontend sends requests to `http://127.0.0.1:8000/predict`.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Welcome message |
| GET | `/health` | Health check |
| POST | `/predict` | Classify an uploaded image |

### POST /predict

- Parameter: `file` (required), an image in JPG, PNG, or WEBP format

Example request:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@image.jpg"
```

Example response:

```json
{
  "class": "cat",
  "confidence": 0.9876
}
```

## Model Architecture

### Custom CNN

- Input: RGB images resized to 128 x 128 pixels
- Feature extractor: 3 convolutional blocks (Conv2D, BatchNorm, ReLU, MaxPool2d)
- Classifier: fully connected layers with dropout
- Output: 2 classes (cat, dog)
- Training: 5 epochs with the Adam optimizer

### ResNet-18

An 18-layer residual network whose skip connections make deeper feature learning more stable. It was used as a stronger baseline and produced the best results in this project.

## Experiment Tracking

Training runs are tracked with MLflow, hosted on a remote server using Docker and Docker Compose:

```bash
docker compose up -d
```

Parameters, metrics (accuracy, precision, recall), and model artifacts are logged for each run so models can be compared and reproduced.

## Tech Stack

- Deep learning: PyTorch, torchvision
- API: FastAPI, Uvicorn
- Frontend: HTML5, CSS3, JavaScript (Fetch API)
- Data handling: gdown, Pillow
- Configuration: PyYAML, python-box
- Utilities: ensure, joblib
- Experiment tracking and deployment: MLflow, Docker, Docker Compose
- Environment: uv, Python 3.12

## Future Improvements

- Add data augmentation (flips, rotations, color jitter).
- Train the custom CNN for more epochs with a learning rate scheduler and early stopping.
- Detect images that are neither a cat nor a dog.
- Add automated tests and a CI workflow.
- Containerize the API for easier deployment.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset sourced from Google Drive.
- Built with the PyTorch and FastAPI communities' tools.
- Inspired by classic computer vision classification tutorials.
