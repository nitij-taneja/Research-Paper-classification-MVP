# Research Paper Classification and Journal Recommendation

This project automates the evaluation of research papers for publishability and recommends suitable journals or conferences for submission using machine learning and natural language processing techniques.

---

**Version:** 2.0

You can request a customized version or have this system built for your own needs. You can even check your own paper at: [Your Link Here]

See how it works in this video: [Demo Video Link]

> **Note:** This is an MVP (Minimum Viable Product) version. For customization or to develop this into a full product, please contact me directly.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Directory Structure](#directory-structure)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Setup Instructions](#setup-instructions)
- [How to Run](#how-to-run)
- [Outputs](#outputs)
- [Future Work](#future-work)
- [License](#license)
- [Credits](#credits)

---

## Project Overview
This project is designed to streamline the manual and time-consuming process of research paper evaluation and conference selection. It consists of two main tasks:

- **Publishability Classification:** Determines if a paper is "Publishable" or "Non-Publishable" based on quality and coherence.
- **Journal Recommendation:** Suggests a suitable journal or conference for publishable papers, along with a rationale.

---

## Directory Structure
```
Research_Paper_Classification/
├── data/
│   ├── raw/                  # Raw input data
│   │   ├── papers/           # Unlabeled research papers
│   │   ├── reference/        # Benchmark reference papers
│   │   ├── non_publishable/  # Non-publishable papers
│   │   ├── publishable/      # Publishable papers categorized by conference
│   │   ├── cvpr/             # CVPR papers
│   │   ├── neurips/          # NeurIPS papers
│   │   ├── emnlp/            # EMNLP papers
│   │   ├── tmlr/             # TMLR papers
│   │   ├── kdd/              # KDD papers
│   ├── processed/            # Intermediate processed data
│   │   ├── markdown_files/   # Optional Markdown outputs for papers
│   │   ├── papers.csv
│   │   ├── reference.csv
│   │   ├── preprocessed_papers.csv
│   │   ├── preprocessed_reference.csv
│   │   ├── metadata_papers.csv
│   │   ├── metadata_reference.csv
│   ├── task1_prediction/     # Task 1 outputs
│   │   ├── predicted_publishability.csv
├── models/                   # Trained models and artifacts
│   ├── publishability_model.pkl
├── results/                  # Final outputs of Task 2
│   ├── results.csv           # Conference recommendations and rationales
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   ├── conference_distribution.png
├── src/                      # Source code for the project
│   ├── pdf_processing.py     # Handles text extraction and section parsing
│   ├── preprocessing.py      # Cleans and preprocesses text data
│   ├── metadata_generation.py# Generates metadata for analysis
│   ├── classification.py     # Task 1: Publishability classification
│   ├── journal_recommendation.py # Task 2: Conference recommendation
│   ├── requirements.txt      # Python dependencies for src scripts
├── frontend/                 # Web frontend (HTML, CSS, JS, templates)
├── README.md                 # Project documentation
├── requirements.txt          # Project-level Python dependencies

```

---

## Features
- Extracts and preprocesses text from research papers in PDF format
- Cleans and normalizes text for consistent analysis
- Generates metadata for research papers, including:
  - Word count, readability, sentiment analysis, and topic diversity
- Classifies research papers into "Publishable" or "Non-Publishable"
- Recommends journals or conferences for publishable papers with justifications
- Provides global and per-paper visualizations (confusion matrix, feature importance, conference distribution)
- Modern web interface for easy interaction
- **Data Augmentation:** Supports augmentation of research paper data to improve model robustness and handle class imbalance.

---

## Technologies Used
- **Languages:** Python
- **Text Processing:** PyPDF2, pdfplumber, PyMuPDF, pytesseract, spacy, nltk
- **Metadata Analysis:** textstat, SentenceTransformers, gensim
- **Machine Learning:** scikit-learn, joblib
- **Models:** Random Forest for publishability classification, Sentence Transformers for embeddings
- **Frontend:** HTML5, CSS3, JavaScript, Bootstrap
- **Deployment:** Flask, Docker, Render (cloud)
- **Data Augmentation:** Custom scripts for augmenting text data (see `src/data_augmentation.py` if present)

---

## Setup Instructions
1. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```
2. **Place Raw Datasets**
   - Put raw PDF files in the `data/raw/` directory under appropriate subdirectories:
     - `papers/` for unlabeled research papers
     - `reference/` for benchmark reference papers

---

## How to Run

### 1. PDF Processing
Extract and structure text from PDFs:
```sh
python src/pdf_processing.py
```

### 2. Preprocessing
Clean and preprocess extracted text:
```sh
python src/preprocessing.py
```

### 3. Metadata Generation
Generate metadata for processed papers:
```sh
python src/metadata_generation.py
```

### 4. Task 1 - Publishability Classification
Train a model and predict publishability:
```sh
python src/classification.py
```

### 5. Task 2 - Journal Recommendation
Recommend journals for publishable papers:
```sh
python src/journal_recommendation.py
```

### 6. Web Application
Run the Flask web app for interactive use:
```sh
python src/app.py
```

### (Optional) Data Augmentation
Augment research paper data to improve training:
```sh
python src/data_augmentation.py
```

---

## Outputs
- **Task 1 - Publishability Predictions:**
  - Saved in `data/task1_prediction/predicted_publishability.csv`
- **Task 2 - Journal Recommendations:**
  - Saved in `results/results.csv`
- **Visualizations:**
  - Confusion Matrix: `results/confusion_matrix.png`
  - Feature Importance: `results/feature_importance.png`
  - Conference Distribution: `results/conference_distribution.png`

---

## Future Work
- Support for multilingual research papers
- Improved feature extraction using more advanced NLP models
- Real-time PDF ingestion and processing via APIs
- Enhanced explainability and interpretability
- Integration with academic databases and APIs

---

## License
This project is licensed under the terms described in the `LICENSE` file (to be added in the repository root).

---

## Credits
- Developed by Nitij Taneja
- Built with Flask, Pandas, Scikit-learn, NLTK, SpaCy, Bootstrap
- Contributions welcome! Open an issue or pull request.

---
