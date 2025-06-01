import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textstat import flesch_reading_ease
import pandas as pd
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
from nltk.tokenize import word_tokenize
import logging
from tqdm import tqdm
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("metadata_generation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("metadata_generator")

# Download necessary NLTK resources with error handling
def download_nltk_resources():
    """Download required NLTK resources with proper error handling."""
    resources = ['punkt', 'vader_lexicon']
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
            logger.info(f"Successfully downloaded NLTK resource: {resource}")
        except Exception as e:
            logger.error(f"Failed to download NLTK resource {resource}: {str(e)}")
            raise

try:
    # Initialize Sentiment Analyzer
    download_nltk_resources()
    sia = SentimentIntensityAnalyzer()
    logger.info("Successfully initialized SentimentIntensityAnalyzer")
except Exception as e:
    logger.error(f"Failed to initialize SentimentIntensityAnalyzer: {str(e)}")
    raise

def calculate_word_count(text):
    """Calculate the word count of a text."""
    if not isinstance(text, str) or not text.strip():
        return 0
    return len(word_tokenize(text))

def calculate_readability(text):
    """Calculate the Flesch Reading Ease score."""
    if not isinstance(text, str) or not text.strip():
        return 0
    try:
        return flesch_reading_ease(text)
    except Exception as e:
        logger.warning(f"Error calculating readability: {str(e)}")
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
        lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)

        # Calculate diversity scores
        diversity_scores = []
        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip() and i < len(tokenized_texts):
                tokens = tokenized_texts[i]
                bow = dictionary.doc2bow(tokens)
                topics = lda_model.get_document_topics(bow)
                # Higher entropy = more diverse topics
                score = sum([topic[1] for topic in topics])
                diversity_scores.append(score)
            else:
                diversity_scores.append(0)

        return diversity_scores
    except Exception as e:
        logger.error(f"Error in topic diversity calculation: {str(e)}")
        return [0] * len(texts)

def calculate_sentiment(text):
    """Calculate the sentiment polarity score of a text."""
    if not isinstance(text, str) or not text.strip():
        return 0
    try:
        sentiment = sia.polarity_scores(text)
        return sentiment['compound']  # Compound score represents overall sentiment
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
    sections = ['abstract', 'introduction', 'methodology', 'results', 'conclusion']
    section_lengths = [calculate_word_count(row.get(section, "")) for section in sections]

    # If all sections are empty, return 0
    if sum(section_lengths) == 0:
        return 0

    # Calculate the coefficient of variation (lower is more balanced)
    import numpy as np
    mean = np.mean(section_lengths)
    std = np.std(section_lengths)
    cv = std / mean if mean > 0 else 0

    # Convert to a balance score (1 = perfectly balanced, 0 = completely unbalanced)
    balance = max(0, 1 - min(cv, 1))
    return balance

def generate_metadata(input_csv, output_csv, keywords=None, progress_callback=None):
    """Generate metadata for each paper."""
    if keywords is None:
        keywords = ['method', 'result', 'study', 'analysis', 'experiment']

    try:
        # Check if input file exists
        if not os.path.exists(input_csv):
            error_msg = f"Input file not found: {input_csv}"
            logger.error(error_msg)
            if progress_callback:
                progress_callback(error_msg)
            return None

        # Load the data
        df = pd.read_csv(input_csv)

        if progress_callback:
            progress_callback(f"Generating metadata for {len(df)} papers...")

        logger.info(f"Generating metadata for {input_csv}...")

        # Generate metadata for each section
        sections = ['abstract', 'introduction', 'methodology', 'results', 'conclusion']
        for section in sections:
            if section in df.columns:
                logger.info(f"Processing {section} section...")
                if progress_callback:
                    progress_callback(f"Processing {section} section...")

                # Apply metrics
                df[f'{section}_word_count'] = df[section].apply(calculate_word_count)
                df[f'{section}_readability'] = df[section].apply(calculate_readability)
                df[f'{section}_sentiment'] = df[section].apply(calculate_sentiment)
                df[f'{section}_keyword_density'] = df[section].apply(lambda text: calculate_keyword_density(text, keywords))

        # Calculate topic diversity for the abstract only
        logger.info("Calculating topic diversity (based on abstracts)...")
        if progress_callback:
            progress_callback("Calculating topic diversity...")

        if 'abstract' in df.columns:
            df['abstract_topic_diversity'] = calculate_topic_diversity(df['abstract'].tolist())

        # Calculate section balance
        logger.info("Calculating section balance...")
        if progress_callback:
            progress_callback("Calculating section balance...")

        df['section_balance'] = df.apply(calculate_section_balance, axis=1)

        # Save metadata
        df.to_csv(output_csv, index=False)
        logger.info(f"Metadata saved to {output_csv}")

        if progress_callback:
            progress_callback(f"Metadata saved to {output_csv}")

        return df

    except Exception as e:
        error_msg = f"Error generating metadata: {str(e)}"
        logger.error(error_msg)
        if progress_callback:
            progress_callback(error_msg)
        raise

if __name__ == "__main__":
    # Define keywords for density calculation
    research_keywords = ['method', 'result', 'study', 'analysis', 'experiment',
                         'data', 'model', 'algorithm', 'performance', 'evaluation']

    # Paths to input and output CSV files
    input_files = [
        ("data/processed/preprocessed_papers.csv", "data/processed/metadata_papers.csv"),
        ("data/processed/preprocessed_reference.csv", "data/processed/metadata_reference.csv")
    ]

    for input_csv, output_csv in input_files:
        generate_metadata(input_csv, output_csv, keywords=research_keywords)
