#  Multimodal Image + Text Classifier

A deep learning project that combines **Computer Vision** and **Natural Language Processing (NLP)** to classify fashion products using both product images and product titles.

The system leverages **OpenAI CLIP (Contrastive Language-Image Pretraining)** to generate multimodal embeddings and predict product categories with high accuracy.

---

##  Features

* Multimodal Learning (Image + Text)
* CLIP-based Feature Extraction
* Fashion Product Classification
* Confidence Score Prediction
* Top-3 Category Predictions
* GPU Accelerated Training (CUDA)
* Interactive Streamlit Dashboard
* FastAPI Backend Support
* Evaluation Metrics & Confusion Matrix
* Production-Ready Inference Pipeline

---

##  Project Architecture

```text
Product Image
       +
Product Title
       |
       v
+-------------------+
|  CLIP Encoder     |
| (Image + Text)    |
+-------------------+
       |
       v
Feature Fusion
       |
       v
Classifier Head
       |
       v
Predicted Category
```

---

##  Project Structure

```text
Multimodal-Image-Text-Classifier/
│
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── inference.py
│   └── api.py
│
├── models/
│   └── best_model.pth
│
├── data/
│   └── styles.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

##  Dataset

Dataset: Fashion Product Images Dataset

The dataset contains:

* Product Images
* Product Titles
* Product Categories

### Categories Used

* Apparel
* Accessories
* Footwear
* Personal Care
* Free Items

Rare categories with insufficient samples were filtered during preprocessing to improve model reliability and evaluation quality.

---

##  Model

### Backbone

**OpenAI CLIP ViT-B/32**

CLIP generates:

* Image Embeddings
* Text Embeddings

These embeddings are concatenated and passed through a custom neural network classifier.

### Classifier Architecture

```text
Image Embedding (512)
         +
Text Embedding (512)
         |
         v
Concatenation (1024)
         |
         v
Dense Layer (512)
         |
         v
Dense Layer (256)
         |
         v
Output Layer
```

---

##  Performance

### Validation Results

| Metric            | Score                                 |
| ----------------- | ------------------------------------- |
| Accuracy          | 99%+                                  |
| Weighted F1 Score | 99%+                                  |
| Macro F1 Score    | High Performance Across Major Classes |

### Evaluation

* Accuracy
* Precision
* Recall
* F1 Score
* Classification Report
* Confusion Matrix

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/tanmaypanda47/Multimodal-Image-Text-Classifier.git

cd Multimodal-Image-Text-Classifier
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

##  Training

Train the model:

```bash
python -m src.train
```

Model checkpoints are saved in:

```text
models/best_model.pth
```

---

##  Evaluation

Run evaluation:

```bash
python -m src.evaluate
```

Outputs:

* Classification Report
* Confusion Matrix
* Accuracy Score
* F1 Score

---

##  Inference

Run prediction:

```bash
python -m src.inference
```

Example Output:

```text
Category: Footwear
Confidence: 98.66%
```

---

##  FastAPI Backend

Start API server:

```bash
uvicorn src.api:app --reload
```

Open Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

##  Streamlit Dashboard

Launch application:

```bash
streamlit run app/streamlit_app.py
```

Dashboard Features:

* Upload Product Image
* Enter Product Title
* Predict Category
* Confidence Visualization
* Top-3 Predictions
* Interactive Charts

---

##  Technologies Used

### Machine Learning

* PyTorch
* Transformers
* OpenAI CLIP

### Data Processing

* Pandas
* NumPy
* Pillow

### Evaluation

* Scikit-Learn
* Matplotlib
* Seaborn

### Deployment

* FastAPI
* Streamlit

---

##  Hardware

Training performed using:

* NVIDIA GeForce RTX 3050 Laptop GPU
* CUDA Enabled PyTorch
* GPU Accelerated Inference

---

##  Learning Outcomes

This project demonstrates:

* Multimodal Deep Learning
* Computer Vision
* Natural Language Processing
* Transformer Models
* CLIP Architecture
* Model Evaluation
* API Development
* Streamlit Deployment
* End-to-End Machine Learning Workflow

---

## 👨‍💻 Author

**Tanmay Panda**

GitHub:
https://github.com/tanmaypanda47

---

##  Future Improvements

* Model Explainability
* Grad-CAM Visualizations
* Docker Deployment
* Cloud Deployment
* Larger Fashion Taxonomy
* Real-Time API Hosting
* Batch Predictions

---

If you found this project useful, consider giving it a ⭐ on GitHub.
