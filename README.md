# Cat vs Dog Classifier

A deep learning-based image classification system that distinguishes between cat and dog images using a Convolutional Neural Network (CNN). The project includes automated data ingestion, model training, evaluation, a FastAPI backend for predictions, and a user-friendly web interface.

## 🎯 Overview

This project implements a complete machine learning pipeline for classifying images as either cats or dogs. It handles everything from downloading the dataset to training a CNN model and serving predictions through a REST API with an interactive frontend.

## 🔧 Features

- **Automated Data Pipeline**: Downloads dataset from Google Drive and extracts it automatically
- **Custom CNN Architecture**: Built with PyTorch featuring convolutional blocks, batch normalization, and dropout layers
- **Model Training & Evaluation**: Trains for 5 epochs with Adam optimizer, evaluates using accuracy, precision, and recall metrics
- **REST API Backend**: FastAPI service with CORS support serving predictions at `/predict` endpoint
- **Interactive Frontend**: Modern web interface with drag-and-drop upload, image preview, and real-time predictions
- **Complete Pipeline Orchestration**: Modular pipeline stages for data ingestion, training, and evaluation
- **Production-Ready**: Includes logging, configuration management, and proper error handling

## 📁 Project Structure

```
cat-vs-dog-classifier/
├── api/                 # FastAPI backend application
│   └── app.py           # Main API with /predict endpoint
├── artifact/            # Generated data and models
│   ├── data_ingestion/  # Downloaded and extracted dataset
│   └── model/           # Trained model.pth file
├── src/Classifier/      # Core ML components
│   ├── components/      # Data ingestion, model definition, training, evaluation
│   ├── config/          # Configuration management (YAML-based)
│   ├── entity/          # Configuration dataclasses
│   ├── pipeline/        # Pipeline orchestration classes
│   └── utils/           # Helper functions (YAML, file operations, logging)
├── templates/           # Frontend web interface
│   └── index.html       # Interactive prediction UI
├── notebooks/           # Jupyter notebooks for research/experimentation
│   └── research.ipynb
├── config/              # Configuration files
│   └── config.yaml      # Pipeline configuration
├── logging/             # Application log files
├── requirements.txt     # Python dependencies
└── main.py              # Pipeline execution script
```

## 🚀 Getting Started

### Prerequisites
- Python 3.12
- Conda (recommended) or pip
- Internet connection for downloading dataset

### Installation

1. **Clone the repository** (if applicable)
2. **Create and activate conda environment**:
   ```bash
   conda create -p env python==3.12 -y
   conda activate env/
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

execute pipeline:
```bash
python main.py
```

### Starting the Services

1. **Start the FastAPI backend**:
   ```bash
   uvicorn api.app:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`

2. **Use the frontend**:
   - Open `templates/index.html` in your web browser
   - The frontend automatically connects to the backend at `http://127.0.0.1:8000/predict`
   - Upload an image to get cat/dog predictions with confidence scores

## 📖 API Documentation

### Endpoints

- `GET /` - Returns welcome message
- `GET /health` - Health check endpoint
- `POST /predict` - Image prediction endpoint

**POST /predict**
- **Parameters**: `file` (required) - Image file (JPG, PNG, WEBP) any type
- **Returns**: JSON with prediction results
  ```json
  {
    "class": "cat",      // or "dog"
    "confidence": 0.9876 // float between 0 and 1
  }
  ```

## 🛠️ Technologies Used

- **Deep Learning Framework**: PyTorch, torchvision
- **API Framework**: FastAPI, Uvicorn
- **Frontend**: HTML5, CSS3, JavaScript (Fetch API)
- **Data Handling**: gdown (Google Drive download), PIL
- **Configuration**: PyYAML, python-box
- **Utilities**: ensure (type checking), joblib
- **Environment**: conda, Python 3.12

## 📊 Model Architecture

The CNN consists of:
- **Feature Extractor**: 3 convolutional blocks (Conv2D → BatchNorm → ReLU → MaxPool2d)
- **Classifier**: Fully connected layers with dropout regularization
- **Input**: RGB images resized to 128×128 pixels
- **Output**: 2-class classification (cat/dog)

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Acknowledgments

- Dataset sourced from Google Drive
- Built with PyTorch and FastAPI communities
- Inspired by classic computer vision classification tutorials
