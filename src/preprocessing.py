import pandas as pd
import spacy
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("preprocessor")

# Load SpaCy language model
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("Successfully loaded SpaCy language model")
except Exception as e:
    logger.error(f"Failed to load SpaCy model: {str(e)}")
    logger.info("Attempting to download SpaCy model...")
    try:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
        nlp = spacy.load("en_core_web_sm")
        logger.info("Successfully downloaded and loaded SpaCy model")
    except Exception as download_error:
        logger.error(f"Failed to download SpaCy model: {str(download_error)}")
        raise

def clean_text(text, paper_id=None, section=None):
    """Clean text by removing special characters, numbers, and stop words."""
    if not isinstance(text, str) or not text.strip():
        logger.warning(f"Non-string or empty text encountered in {section} of Paper ID {paper_id}. Skipping...")
        return ""

    try:
        # Process text in chunks if it's very large
        max_chunk_size = 100000  # SpaCy works better with smaller chunks
        if len(text) > max_chunk_size:
            logger.info(f"Large text detected in {section} of Paper ID {paper_id}. Processing in chunks.")
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            processed_chunks = []

            for chunk in chunks:
                doc = nlp(chunk.lower())
                processed_chunks.append(" ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha]))

            return " ".join(processed_chunks)
        else:
            doc = nlp(text.lower())
            return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])
    except Exception as e:
        logger.error(f"Error cleaning text for Paper ID {paper_id}, Section: {section}. Error: {str(e)}")
        # Return a simplified cleaning as fallback
        return " ".join([word.lower() for word in text.split() if len(word) > 2])

def preprocess_text(input_csv, output_csv, progress_callback=None):
    """Clean and preprocess text, then save to a new CSV."""
    try:
        # Load the input CSV
        df = pd.read_csv(input_csv)

        if progress_callback:
            progress_callback(f"Starting text cleaning for {input_csv}...")

        logger.info(f"Starting text cleaning for {input_csv}...")

        # Apply text cleaning with progress tracking
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing"):
            paper_id = row.get('Paper ID', f"Row_{index}")

            if progress_callback and index % 5 == 0:  # Update progress every 5 papers
                progress_callback(f"Processing Paper ID: {paper_id} ({index+1}/{len(df)})")

            logger.info(f"Processing Paper ID: {paper_id}")

            for section in ['abstract', 'introduction', 'methodology', 'results', 'conclusion']:
                if section in df.columns:
                    df.at[index, section] = clean_text(row.get(section, ""), paper_id, section)

        # Save the cleaned data
        df.to_csv(output_csv, index=False)
        logger.info(f"Preprocessed data saved to {output_csv}")

        if progress_callback:
            progress_callback(f"Preprocessed data saved to {output_csv}")

        return df

    except Exception as e:
        logger.error(f"Error in preprocessing: {str(e)}")
        if progress_callback:
            progress_callback(f"Error in preprocessing: {str(e)}")
        raise

if __name__ == "__main__":
    # Paths to input and output CSV files
    input_files = [
        ("data/processed/processed_papers.csv", "data/processed/preprocessed_papers.csv"),
        ("data/processed/processed_reference.csv", "data/processed/preprocessed_reference.csv")
    ]

    for input_csv, output_csv in input_files:
        preprocess_text(input_csv, output_csv)
