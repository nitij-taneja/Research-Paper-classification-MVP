# --- FINAL APP V4 - SELF CONTAINED --- 
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import uuid
import json
import logging
from werkzeug.utils import secure_filename
import threading
import time
import pandas as pd
import numpy as np
import pickle
import joblib  # Import joblib
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import re # Import re for user\"s regex logic

# --- Dependencies for Embedded Metadata Generation --- 
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
# --- End Metadata Dependencies --- 

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("app")

# --- Download NLTK Resources --- 
def download_nltk_resources():
    """Download required NLTK resources with proper error handling."""
    resources = ["punkt", "vader_lexicon"]
    downloaded_all = True
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully ensured NLTK resource: {resource}")
        except Exception as e:
            logger.error(f"Failed to download NLTK resource {resource}: {str(e)}. Please ensure NLTK data is available.")
            downloaded_all = False
    return downloaded_all

# Attempt to download NLTK data and initialize SIA
sia = None
if download_nltk_resources():
    try:
        sia = SentimentIntensityAnalyzer()
        logger.info("Successfully initialized SentimentIntensityAnalyzer")
    except Exception as e:
        logger.error(f"Failed to initialize SentimentIntensityAnalyzer even after download attempt: {str(e)}")
else:
    logger.error("Could not download all necessary NLTK resources. Sentiment analysis may fail.")
# --- End NLTK Setup --- 

# Initialize Flask app
app = Flask(__name__, static_folder="../frontend/static", template_folder="../frontend/templates")
CORS(app)

# Get base directory for reliable path resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuration
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "..", "data", "raw", "papers")
app.config["RESULTS_FOLDER"] = os.path.join(BASE_DIR, "..", "results")
app.config["MODELS_FOLDER"] = os.path.join(BASE_DIR, "..", "models")
app.config["PROCESSED_FOLDER"] = os.path.join(BASE_DIR, "..", "data", "processed")
app.config["TASK1_PREDICTION_FOLDER"] = os.path.join(BASE_DIR, "..", "data", "task1_prediction")
app.config["ALLOWED_EXTENSIONS"] = {"pdf"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload size

# Create necessary directories
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["RESULTS_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODELS_FOLDER"], exist_ok=True)
os.makedirs(app.config["PROCESSED_FOLDER"], exist_ok=True)
os.makedirs(app.config["TASK1_PREDICTION_FOLDER"], exist_ok=True)

# Store processing status
processing_status = {}

# Initialize vectorizer globally (trained on reference abstracts)
vectorizer = None
published_abstracts = ""
unpublished_abstracts = ""

def compute_similarity(text_1, text_2):
    """Compute text similarity using TF-IDF vectors."""
    global vectorizer
    if not vectorizer or not text_1 or not text_2:
        return 0
    try:
        vec_1 = vectorizer.transform([text_1]).toarray()
        vec_2 = vectorizer.transform([text_2]).toarray()
        return cosine_similarity(vec_1, vec_2)[0][0]
    except Exception:
        return 0

def extract_advanced_features(row):
    """Extract advanced features used during training."""
    global published_abstracts, unpublished_abstracts

    try:
        abstract = row.get("abstract", "")
        abstract_sim_published = compute_similarity(abstract, published_abstracts)
        abstract_sim_unpublished = compute_similarity(abstract, unpublished_abstracts)
        similarity_ratio = abstract_sim_published / (abstract_sim_unpublished + 1e-6)

        word_count_diff_published = abs(row.get('abstract_word_count', 0) - published_word_count_mean)
        word_count_diff_unpublished = abs(row.get('abstract_word_count', 0) - unpublished_word_count_mean)

        features = {
            'abstract_sim_published': abstract_sim_published,
            'abstract_sim_unpublished': abstract_sim_unpublished,
            'similarity_ratio': similarity_ratio,
            'word_count_diff_published': word_count_diff_published,
            'word_count_diff_unpublished': word_count_diff_unpublished
        }
        return pd.Series(features)
    except Exception as e:
        logger.error(f"Error extracting advanced features: {str(e)}")
        return pd.Series({
            'abstract_sim_published': 0,
            'abstract_sim_unpublished': 0,
            'similarity_ratio': 0,
            'word_count_diff_published': 0,
            'word_count_diff_unpublished': 0
        })

# --- PDF Processing Functions (embedded) --- 
def extract_text_pypdf2(file_path):
    """Extract raw text from a PDF using PyPDF2."""
    try:
        with open(file_path, "rb") as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PyPDF2 failed for {file_path}: {str(e)}")
        return ""

def extract_text_pdfplumber(file_path):
    """Extract raw text from a PDF using pdfplumber."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"pdfplumber failed for {file_path}: {str(e)}")
        return ""

def extract_text_pymupdf(file_path):
    """Extract raw text from a PDF using PyMuPDF."""
    try:
        text = ""
        with fitz.open(file_path) as pdf:
            for page in pdf:
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"PyMuPDF failed for {file_path}: {str(e)}")
        return ""

def extract_text_with_ocr(file_path, progress_callback=None):
    """Fallback: Extract text using OCR from PDF images."""
    try:
        text = ""
        with fitz.open(file_path) as pdf:
            total_pages = len(pdf)
            for page_num, page in enumerate(pdf):
                if progress_callback:
                    progress_callback(f"OCR processing page {page_num+1}/{total_pages}")

                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Higher resolution for better OCR
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img)
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"OCR extraction failed for {file_path}: {str(e)}")
        return ""

def clean_text(text):
    """Remove repeated lines and normalize whitespace (User's version)."""
    if not text:
        return ""

    lines = text.split("\n")
    unique_lines = []
    seen_lines = set()

    for line in lines:
        line = line.strip()
        # Basic duplicate removal and stripping.
        if line and line not in seen_lines:
            unique_lines.append(line)
            seen_lines.add(line)

    # Replace problematic characters (User's version)
    cleaned_text = "\n".join(unique_lines)
    cleaned_text = cleaned_text.replace("\"", "").replace(",", " ") 
    # Add any other specific regex replacements from user's app.py here if needed

    return cleaned_text.strip()

def extract_text_combined(file_path, progress_callback=None):
    """Combine text extraction results and apply user's cleaning logic."""
    extraction_methods = [
        ("PyPDF2", extract_text_pypdf2),
        ("PDFPlumber", extract_text_pdfplumber),
        ("PyMuPDF", extract_text_pymupdf)
    ]

    results = {}
    success = False

    for method_name, method_func in extraction_methods:
        if progress_callback:
            progress_callback(f"Trying {method_name}...")

        try:
            text = method_func(file_path)
            if text and len(text.strip()) > 100:  # Minimum viable text length
                results[method_name] = text
                success = True
                logger.info(f"{method_name} successfully extracted {len(text)} characters from {file_path}")
        except Exception as e:
            logger.error(f"{method_name} failed: {str(e)}")
            if progress_callback:
                progress_callback(f"{method_name} failed: {str(e)}")

    # If all methods fail, try OCR with progress updates
    if not success:
        logger.warning(f"All standard methods failed for {file_path}. Attempting OCR extraction...")
        if progress_callback:
            progress_callback("All standard methods failed. Attempting OCR extraction...")
        try:
            ocr_text = extract_text_with_ocr(file_path, progress_callback)
            if ocr_text:
                results["OCR"] = ocr_text
                success = True
                logger.info(f"OCR successfully extracted {len(ocr_text)} characters from {file_path}")
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            if progress_callback:
                progress_callback(f"OCR extraction failed: {str(e)}")

    # Combine and clean results using USER'S clean_text function
    if success:
        merged_text = "\n".join([text for text in results.values() if text.strip()])
        return clean_text(merged_text) # Apply user's cleaning
    else:
        logger.error(f"All extraction methods failed for {file_path}")
        return ""

def parse_sections(text):
    """Parse the text into sections using User's logic/keywords if provided."""
    # Using the robust version from previous iterations, adaptable if user provided specific regex.
    sections = {
        "abstract": "",
        "introduction": "",
        "methodology": "",
        "results": "",
        "conclusion": ""
    }

    if not text:
        return sections

    text_lower = text.lower()

    # Define section keywords and their alternatives (Corrected formatting)
    section_keywords = {
        "abstract": ["abstract", "summary"],
        "introduction": ["introduction", "background", "overview"],
        "methodology": ["methodology", "methods", "approach", "experimental setup", "materials and methods"],
        "results": ["results", "findings", "evaluation", "experiments", "experimental results"],
        "conclusion": ["conclusion", "conclusions", "discussion", "future work", "summary and conclusion"]
    }

    # Find all potential section starts
    section_positions = {}
    for section, keywords in section_keywords.items():
        for keyword in keywords:
            pos = text_lower.find(keyword)
            if pos != -1:
                # Check if it's likely a section header (preceded by newline or beginning of text)
                if pos == 0 or text_lower[pos-1] in ["\n", " ", "\t"]:
                    # Check if the keyword is followed by a newline or end of line reasonably soon
                    line_end_pos = text_lower.find("\n", pos)
                    if line_end_pos == -1 or (line_end_pos - pos) < 50: # Header shouldn't be too long
                         section_positions[section] = pos
                         break

    # Sort sections by their position in the text
    sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])

    # Extract text between sections
    for i, (section, start_pos) in enumerate(sorted_sections):
        # Find the end of the current section (start of the next section or end of text)
        if i < len(sorted_sections) - 1:
            end_pos = sorted_sections[i+1][1]
        else:
            end_pos = len(text)

        # Extract the section text, trying to skip the header line itself
        header_line_end = text.find("\n", start_pos)
        content_start = header_line_end + 1 if header_line_end != -1 else start_pos + len(section_keywords[section][0])
        
        section_text = text[content_start:end_pos].strip()
        sections[section] = section_text

    # Log if any section is empty
    for key, value in sections.items():
        if not value.strip():
            logger.warning(f'"{key}" section is empty or poorly extracted.')

    return sections
# --- End of PDF Processing Functions ---

# --- Embedded Metadata Generation Functions --- 
def calculate_word_count(text):
    """Calculate the word count of a text."""
    if not isinstance(text, str) or not text.strip():
        return 0
    try:
        return len(word_tokenize(text))
    except Exception as e:
        logger.warning(f"NLTK word_tokenize failed: {e}. Returning 0 word count.")
        return 0

def calculate_readability(text):
    """Calculate the Flesch Reading Ease score."""
    if not isinstance(text, str) or not text.strip():
        return 0
    try:
        return flesch_reading_ease(text)
    except Exception as e:
        logger.warning(f"Error calculating readability with textstat: {str(e)}. Returning 0.")
        return 0

def calculate_topic_diversity(texts, num_topics=5):
    """Calculate topic diversity using LDA."""
    # Filter out non-string or empty texts
    valid_texts = [text for text in texts if isinstance(text, str) and text.strip()]

    if not valid_texts:
        logger.warning("No valid texts for topic diversity calculation")
        return [0] * len(texts)

    try:
        # Tokenize texts
        tokenized_texts = [word_tokenize(text) for text in valid_texts]

        # Filter out very short texts
        tokenized_texts = [tokens for tokens in tokenized_texts if len(tokens) > 10]

        if not tokenized_texts:
            logger.warning("No valid tokenized texts for topic diversity calculation")
            return [0] * len(texts)

        # Create a dictionary and corpus for LDA
        dictionary = Dictionary(tokenized_texts)
        corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

        # Train LDA model
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10, random_state=42)

        # Calculate diversity scores
        diversity_scores_map = {}
        for i, text in enumerate(valid_texts):
             tokens = tokenized_texts[i]
             bow = dictionary.doc2bow(tokens)
             topics = lda_model.get_document_topics(bow, minimum_probability=0.01)
             # Higher entropy = more diverse topics (using sum of probabilities as proxy)
             score = sum([topic[1] for topic in topics])
             diversity_scores_map[i] = score

        # Map scores back to original texts list
        final_scores = [diversity_scores_map.get(valid_texts.index(text), 0) if isinstance(text, str) and text.strip() else 0 for text in texts]
        return final_scores

    except ImportError:
        logger.error("Gensim library not found. Cannot calculate topic diversity. Please install gensim.")
        return [0] * len(texts)
    except Exception as e:
        logger.error(f"Error in topic diversity calculation: {str(e)}")
        return [0] * len(texts)

def calculate_sentiment(text):
    """Calculate the sentiment polarity score of a text."""
    if not isinstance(text, str) or not text.strip():
        return 0
    if sia is None:
        logger.warning("SentimentIntensityAnalyzer not initialized. Returning 0 sentiment.")
        return 0
    try:
        sentiment = sia.polarity_scores(text)
        return sentiment["compound"]  # Compound score represents overall sentiment
    except Exception as e:
        logger.warning(f"Error calculating sentiment: {str(e)}")
        return 0

def calculate_keyword_density(text, keywords):
    """Calculate the density of given keywords in a text."""
    if not isinstance(text, str) or not text.strip():
        return 0
    try:
        text_tokens = word_tokenize(text.lower())
        if not text_tokens:
            return 0

        keyword_count = sum(text_tokens.count(keyword.lower()) for keyword in keywords)
        return keyword_count / len(text_tokens)
    except Exception as e:
        logger.warning(f"Error calculating keyword density: {str(e)}")
        return 0

def calculate_section_balance(row):
    """Calculate how balanced the paper sections are."""
    sections = ["abstract", "introduction", "methodology", "results", "conclusion"]
    section_lengths = [calculate_word_count(row.get(section, "")) for section in sections]

    # If all sections are empty, return 0
    if sum(section_lengths) == 0:
        return 0

    # Calculate the coefficient of variation (lower is more balanced)
    mean = np.mean(section_lengths)
    std = np.std(section_lengths)
    cv = std / mean if mean > 0 else 0

    # Convert to a balance score (1 = perfectly balanced, 0 = completely unbalanced)
    balance = max(0, 1 - min(cv, 1))
    return balance

def generate_metadata_embedded(input_df, keywords=None, progress_callback=None):
    """Generate metadata for each paper (Embedded Version). Takes DataFrame as input."""
    if keywords is None:
        keywords = ["method", "result", "study", "analysis", "experiment"]

    try:
        df = input_df.copy() # Work on a copy

        if progress_callback:
            progress_callback(f"Generating metadata for {len(df)} entries...")

        logger.info(f"Generating metadata for DataFrame...")

        # Generate metadata for each section
        sections = ["abstract", "introduction", "methodology", "results", "conclusion"]
        for section in sections:
            if section in df.columns:
                logger.info(f"Processing {section} section...")
                if progress_callback:
                    progress_callback(f"Processing {section} section...")

                # Apply metrics safely
                df[f"{section}_word_count"] = df[section].apply(calculate_word_count)
                df[f"{section}_readability"] = df[section].apply(calculate_readability)
                df[f"{section}_sentiment"] = df[section].apply(calculate_sentiment)
                df[f"{section}_keyword_density"] = df[section].apply(lambda text: calculate_keyword_density(text, keywords))
            else:
                 logger.warning(f'Column "{section}" not found in DataFrame for metadata generation.')

        # Calculate topic diversity for the abstract only
        logger.info("Calculating topic diversity (based on abstracts)...")
        if progress_callback:
            progress_callback("Calculating topic diversity...")

        if "abstract" in df.columns:
            df["abstract_topic_diversity"] = calculate_topic_diversity(df["abstract"].tolist())
        else:
            df["abstract_topic_diversity"] = 0 # Assign default if abstract column missing
            logger.warning('Column "abstract" not found for topic diversity calculation.')

        # Calculate section balance
        logger.info("Calculating section balance...")
        if progress_callback:
            progress_callback("Calculating section balance...")

        df["section_balance"] = df.apply(calculate_section_balance, axis=1)

        logger.info(f"Metadata generation complete for DataFrame.")

        if progress_callback:
            progress_callback(f"Metadata generation complete.")

        return df

    except Exception as e:
        error_msg = f"Error generating metadata: {str(e)}"
        logger.error(error_msg)
        if progress_callback:
            progress_callback(error_msg)
        raise
# --- End of Embedded Metadata Generation --- 

# --- Self-contained Journal Recommendation Functions ---
def recommend_journal_self_contained(paper_features, reference_data_path):
    """Self-contained journal recommendation function that doesn't rely on external imports."""
    try:
        # Check if reference data exists
        if not os.path.exists(reference_data_path):
            logger.warning(f"Reference metadata file not found at: {reference_data_path}")
            return "General Conference", "No reference data available for specific recommendations."
        
        # Load reference data (keep all columns for lookup)
        reference_data_full = pd.read_csv(reference_data_path)
        
        # Get feature columns (exclude non-feature columns)
        non_feature_cols = [
            "Paper ID", "Label", "Conference",
            "abstract", "introduction", "methodology", "results", "conclusion",
            "rationale", "publishable", "confidence"
        ]
        feature_cols = [col for col in reference_data_full.columns if col not in non_feature_cols]
        # Filter only numeric columns for similarity
        reference_data_numeric = reference_data_full[feature_cols].apply(pd.to_numeric, errors='coerce')
        paper_features_numeric = paper_features[feature_cols].apply(pd.to_numeric, errors='coerce')

        # Ensure paper_features has the same columns
        common_cols = [col for col in feature_cols if col in paper_features_numeric.columns]
        if not common_cols:
            logger.warning("Feature mismatch between paper and reference data for recommendation.")
            return "General Conference", "Feature mismatch between paper and reference data."
        
        # Calculate similarity
        paper_vector = paper_features_numeric[common_cols].fillna(0).values
        reference_vectors = reference_data_numeric[common_cols].fillna(0).values
        
        # Calculate cosine similarity
        similarities = cosine_similarity(paper_vector, reference_vectors)[0]
        
        # Get top 3 most similar papers
        top_indices = np.argsort(similarities)[-3:][::-1]
        
        # Get conferences of top papers (lookup in full reference data)
        top_conferences = []
        for idx in top_indices:
            if "Conference" in reference_data_full.columns and idx < len(reference_data_full):
                conf = reference_data_full.iloc[idx]["Conference"]
                if isinstance(conf, str) and conf.strip():
                    top_conferences.append(conf)
        
        # Count conference occurrences
        if top_conferences:
            from collections import Counter
            conference_counts = Counter(top_conferences)
            recommended_conference = conference_counts.most_common(1)[0][0]
            rationale = f"Based on content similarity with published papers in our database, this paper is most suitable for {recommended_conference}."
            return recommended_conference, rationale
        else:
            logger.warning("No suitable conferences found among top similar papers.")
            return "General Conference", "Based on limited analysis, a general conference is recommended."
        
    except Exception as e:
        logger.error(f"Error in self-contained journal recommendation: {str(e)}")
        return "General Conference", f"Error in recommendation process: {str(e)}"
# --- End of Journal Recommendation Functions ---

# --- Safe model loading function --- 
def safe_load_model(model_folder):
    """Safely load a pickled model, prioritizing joblib, with error handling."""
    model_paths_to_try = [
        # Prioritize joblib
        (os.path.join(model_folder, "publishability_model.joblib"), joblib.load),
        # Fallback to pickle
        (os.path.join(model_folder, "publishability_model.pkl"), pickle.load),
        (os.path.join(model_folder, "enhanced_model.pkl"), pickle.load),
        (os.path.join(model_folder, "model.pkl"), pickle.load)
    ]
    
    model = None
    last_error = "No model files found."
    
    for model_path, load_func in model_paths_to_try:
        if os.path.exists(model_path):
            try:
                logger.info(f"Attempting to load model from: {model_path} using {load_func.__name__}")
                if load_func == pickle.load:
                    with open(model_path, "rb") as f:
                        model = load_func(f)
                else: # joblib.load
                    model = load_func(model_path)
                logger.info(f"Model loaded successfully from {model_path}")
                return model, None  # Return successfully loaded model
            except Exception as e:
                last_error = f"Error loading model from {model_path}: {str(e)}"
                logger.error(last_error)
                # Continue to try the next model file
        else:
            logger.info(f"Model file not found: {model_path}")
            
    return None, last_error # Return None if all attempts fail
# --- End of Safe Model Loading --- 

def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

def process_paper_inference_only(file_path, submission_id):
    """Process a paper using pre-trained models for inference only."""
    try:
        # Update status to processing
        processing_status[submission_id] = {
            "status": "processing",
            "progress": 10,
            "message": "Starting processing..."
        }
        save_status(submission_id)
        
        # Extract paper ID from filename
        paper_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Step 1: Extract text from PDF
        processing_status[submission_id] = {
            "status": "processing",
            "progress": 20,
            "message": "Extracting text from PDF..."
        }
        save_status(submission_id)
        
        # Extract text from PDF using embedded functions (with user's cleaning)
        def progress_callback(message):
            processing_status[submission_id]["message"] = message
            save_status(submission_id)
            
        raw_text = extract_text_combined(file_path, progress_callback)
        if not raw_text.strip():
            raise Exception("Failed to extract text from PDF")
        
        # Parse sections (using potentially user-updated logic)
        sections = parse_sections(raw_text)
        
        # Create a DataFrame with the extracted text
        extracted_df = pd.DataFrame({
            "Paper ID": [paper_id],
            "abstract": [sections.get("abstract", "")],
            "introduction": [sections.get("introduction", "")],
            "methodology": [sections.get("methodology", "")],
            "results": [sections.get("results", "")], # Added results section
            "conclusion": [sections.get("conclusion", "")]
        })
        
        # Step 2: Preprocess text (Optional - can be skipped if metadata uses raw sections)
        # For simplicity in self-contained, we will generate metadata from extracted_df directly
        processing_status[submission_id] = {
            "status": "processing",
            "progress": 40,
            "message": "Skipping separate preprocessing step..."
        }
        save_status(submission_id)
        preprocessed_df = extracted_df # Use extracted directly
        
        # Step 3: Generate metadata using EMBEDDED function
        processing_status[submission_id] = {
            "status": "processing",
            "progress": 60,
            "message": "Generating metadata..."
        }
        save_status(submission_id)
        
        try:
            # Call the embedded metadata generation function
            paper_data_with_metadata = generate_metadata_embedded(preprocessed_df, progress_callback=progress_callback)
            # Save the generated metadata (optional, for debugging)
            metadata_csv_path = os.path.join(app.config["PROCESSED_FOLDER"], f"{paper_id}_metadata.csv")
            paper_data_with_metadata.to_csv(metadata_csv_path, index=False)
            logger.info(f"Generated metadata saved to {metadata_csv_path}")
        except Exception as e:
            logger.error(f"Error during embedded metadata generation: {str(e)}")
            raise Exception(f"Failed to generate metadata: {str(e)}")
        
        # Load reference data and fit vectorizer
        reference_path = os.path.join(app.config["PROCESSED_FOLDER"], "metadata_reference.csv")
        reference_df = pd.read_csv(reference_path)

        global vectorizer, published_abstracts, unpublished_abstracts
        vectorizer = TfidfVectorizer(stop_words="english")
        vectorizer.fit(reference_df["abstract"].fillna(""))

        published_abstracts = " ".join(reference_df[reference_df["Label"] == 1]["abstract"].fillna(""))
        unpublished_abstracts = " ".join(reference_df[reference_df["Label"] == 0]["abstract"].fillna(""))

        global published_word_count_mean, unpublished_word_count_mean
        published_word_count_mean = reference_df[reference_df["Label"] == 1]["abstract_word_count"].mean()
        unpublished_word_count_mean = reference_df[reference_df["Label"] == 0]["abstract_word_count"].mean()

        # Apply advanced features
        advanced_features = paper_data_with_metadata.apply(extract_advanced_features, axis=1)

        # Merge them into paper_data_with_metadata
        paper_data = pd.concat([paper_data_with_metadata, advanced_features], axis=1)
        
        # Step 4: Run classification using pre-trained model
        processing_status[submission_id] = {
            "status": "processing",
            "progress": 80,
            "message": "Classifying paper..."
        }
        save_status(submission_id)
        
        # Load model safely (prioritizing joblib)
        model, error_msg = safe_load_model(app.config["MODELS_FOLDER"])
        if not model:
            raise Exception(f"Could not load any classification model. Last error: {error_msg}")
        
        # Use the DataFrame with generated metadata
        
        if not isinstance(paper_data, pd.DataFrame) or paper_data.empty:
             raise Exception("Metadata generation did not produce a valid DataFrame")

        # Prepare features with robust validation
        expected_features = getattr(model, "feature_names_in_", None)
        if expected_features is not None:
            logger.info(f"Model expects features: {expected_features}")
            feature_cols = [col for col in expected_features if col in paper_data.columns]
            missing_expected = [col for col in expected_features if col not in paper_data.columns]
            if missing_expected:
                logger.warning(f"Metadata missing expected model features: {missing_expected}. Proceeding with available features.")
        else:
            logger.warning("Model does not have feature_names_in_ attribute. Using fallback feature selection.")
            # Fallback: Use all generated metadata columns (excluding text sections and ID)
            feature_cols = [col for col in paper_data.columns 
                            if col not in ["Paper ID", "abstract", "introduction", "methodology", "results", "conclusion"]]
        
        if not feature_cols:
             raise Exception("No valid feature columns identified or generated for classification.")
             
        logger.info(f"Using features for prediction: {feature_cols}")
        
        missing_cols = [col for col in feature_cols if col not in paper_data.columns]
        if missing_cols:
            raise Exception(f"Missing required feature columns in generated metadata: {missing_cols}")
        
        try:
            X = paper_data[feature_cols].fillna(0)
            if not isinstance(X, pd.DataFrame) or X.empty:
                 raise Exception("Feature DataFrame X is empty after selection and fillna.")
        except Exception as e:
            raise Exception(f"Error preparing features for classification: {str(e)}")
        
        # Make prediction
        try:
            y_prob = model.predict_proba(X)
            confidence = y_prob[0][1]  # Probability of class 1 (publishable)
        except AttributeError:
             logger.warning(f"Model does not support predict_proba. Using predict output for confidence.")
             y_pred_temp = model.predict(X)
             confidence = float(y_pred_temp[0]) # Use prediction as confidence (0 or 1)
        except Exception as e:
            logger.warning(f"Could not get prediction probability: {str(e)}. Using default confidence.")
            confidence = 0.8  # Default confidence
        
        y_pred = model.predict(X)
        publishable = bool(y_pred[0])
        
        # Save prediction to CSV (optional)
        prediction_df = pd.DataFrame({
            "Paper ID": [paper_id],
            "publishable": [int(publishable)],
            "confidence": [confidence]
        })
        prediction_csv_path = os.path.join(app.config["TASK1_PREDICTION_FOLDER"], f"{paper_id}_prediction.csv")
        prediction_df.to_csv(prediction_csv_path, index=False)
        
        # Step 5: Run journal recommendation if publishable
        conference = "N/A"
        rationale = "N/A"
        
        if publishable:
            processing_status[submission_id] = {
                "status": "processing",
                "progress": 90,
                "message": "Generating journal recommendations..."
            }
            save_status(submission_id)
            
            # Use self-contained recommendation (corrected filename)
            reference_metadata_path = os.path.join(app.config["PROCESSED_FOLDER"], "metadata_reference.csv") # Corrected filename
            conference, rationale = recommend_journal_self_contained(paper_data, reference_metadata_path)
            logger.info(f"Journal recommendation from self-contained function: {conference}")
        
        # Create result object
        result = {
            "paper_id": paper_id,
            "publishable": publishable,
            "confidence": float(confidence),
            "conference": conference,
            "rationale": rationale
        }
        
        # Save result
        result_path = os.path.join(app.config["RESULTS_FOLDER"], f"{submission_id}_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f)
        
        # Update status
        processing_status[submission_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Processing completed successfully",
            "result": result
        }
        save_status(submission_id)
        
        # Clean up temporary files (optional, keep metadata for debugging?)
        # try:
        #     if os.path.exists(metadata_csv_path):
        #          os.remove(metadata_csv_path)
        # except Exception as e:
        #     logger.warning(f"Could not clean up temporary metadata file: {str(e)}")
        #     pass
        
        return True
    except Exception as e:
        logger.exception(f"Error processing paper {submission_id}: {str(e)}") # Log full traceback
        processing_status[submission_id] = {
            "status": "error",
            "progress": 100,
            "message": f"Error: {str(e)}"
        }
        save_status(submission_id)
        return False

def save_status(submission_id):
    """Save processing status to file."""
    status_path = os.path.join(app.config["RESULTS_FOLDER"], f"{submission_id}_status.json")
    try:
        # Ensure status exists before saving
        if submission_id in processing_status:
            with open(status_path, "w") as f:
                json.dump(processing_status[submission_id], f)
        else:
            logger.warning(f"Attempted to save status for non-existent submission ID: {submission_id}")
    except Exception as e:
         logger.error(f"Failed to save status for {submission_id}: {str(e)}")

# --- Flask Routes --- 
@app.route("/")
def index():
    """Render the main page (without visualizations)."""
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_file():
    """Handle file upload."""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    try:
        submission_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        # Ensure filename is unique to avoid overwrites if multiple users upload same name
        # A simple approach (better might involve subdirs or UUID filenames):
        base, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(file_path):
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{base}_{counter}{ext}")
            counter += 1
            
        file.save(file_path)
        logger.info(f"File saved to: {file_path}")
        
        processing_status[submission_id] = {
            "status": "uploaded",
            "progress": 0,
            "message": "File uploaded, waiting for processing"
        }
        save_status(submission_id)
        
        thread = threading.Thread(target=process_paper_inference_only, args=(file_path, submission_id))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "submission_id": submission_id,
            "message": "File uploaded successfully, processing started"
        })
    except Exception as e:
        logger.exception(f"Error uploading file: {str(e)}") # Log full traceback
        return jsonify({"error": "Internal server error during upload."}), 500

@app.route("/status/<submission_id>", methods=["GET"])
def get_status(submission_id):
    """Get processing status."""
    status_path = os.path.join(app.config["RESULTS_FOLDER"], f"{submission_id}_status.json")
    if os.path.exists(status_path):
        try:
            with open(status_path, "r") as f:
                status = json.load(f)
            return jsonify(status)
        except Exception as e:
             logger.error(f"Failed to load status file for {submission_id}: {str(e)}")
    
    if submission_id in processing_status:
        return jsonify(processing_status[submission_id])
    
    return jsonify({"error": "Submission ID not found"}), 404

@app.route("/result/<submission_id>", methods=["GET"])
def get_result(submission_id):
    """Get processing result."""
    result_path = os.path.join(app.config["RESULTS_FOLDER"], f"{submission_id}_result.json")
    if os.path.exists(result_path):
        try:
            with open(result_path, "r") as f:
                result = json.load(f)
            return jsonify(result)
        except Exception as e:
             logger.error(f"Failed to load result file for {submission_id}: {str(e)}")

    if submission_id in processing_status and "result" in processing_status[submission_id]:
        return jsonify(processing_status[submission_id]["result"])
    
    return jsonify({"error": "Result not found or processing incomplete/failed"}), 404

@app.route("/visualizations")
def get_visualizations():
    """Get all global visualizations (for about or documentation page)."""
    visualizations = []
    vis_files = [
        "confusion_matrix.png",
        "feature_importance.png",
        "roc_curve_comparison.png",
        "model_performance_comparison.png",
        "shap_summary.png",
        "conference_distribution.png"
    ]
    for vis_file in vis_files:
        vis_path = os.path.join(app.config["RESULTS_FOLDER"], vis_file)
        if os.path.exists(vis_path):
            visualizations.append({
                "name": vis_file.split(".")[0].replace("_", " ").title(),
                "path": f"/visualization/{vis_file}"
            })
    return jsonify(visualizations)

@app.route("/visualization/<filename>")
def serve_visualization(filename):
    """Serve visualization file."""
    safe_filename = secure_filename(filename)
    return send_from_directory(app.config["RESULTS_FOLDER"], safe_filename)

@app.route("/about")
def about():
    """Render the about page with visualizations."""
    # Get visualizations
    visualizations = []
    vis_files = [
        "confusion_matrix.png",
        "feature_importance.png",
        "roc_curve_comparison.png",
        "model_performance_comparison.png",
        "shap_summary.png",
        "conference_distribution.png"
    ]
    for vis_file in vis_files:
        vis_path = os.path.join(app.config["RESULTS_FOLDER"], vis_file)
        if os.path.exists(vis_path):
            visualizations.append({
                "name": vis_file.split(".")[0].replace("_", " ").title(),
                "path": f"/visualization/{vis_file}"
            })
    return render_template("about.html", visualizations=visualizations)

@app.route("/documentation")
def documentation():
    """Render the documentation page."""
    return render_template("documentation.html")

import os

if __name__ == "__main__":
    if not sia:
        logger.warning("NLTK setup incomplete. Sentiment features might be zero.")
    
    port = int(os.environ.get("PORT", 10000))  # Use Render's port if available
    app.run(host="0.0.0.0", port=port, debug=False)  # Turn off debug mode for stability


