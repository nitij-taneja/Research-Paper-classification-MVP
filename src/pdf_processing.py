import os
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import pytesseract
import logging
from tqdm import tqdm
import concurrent.futures

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("pdf_processor")

def extract_text_pypdf2(file_path):
    """Extract raw text from a PDF using PyPDF2."""
    try:
        with open(file_path, 'rb') as pdf_file:
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
    """Remove repeated lines and normalize whitespace."""
    if not text:
        return ""

    lines = text.split("\n")
    unique_lines = []
    seen_lines = set()

    for line in lines:
        line = line.strip()
        if line and line not in seen_lines:
            unique_lines.append(line)
            seen_lines.add(line)

    # Replace problematic characters
    cleaned_text = "\n".join(unique_lines)
    cleaned_text = cleaned_text.replace('"', "'").replace(",", " ")

    return cleaned_text.strip()


def enhanced_extract_text_combined(file_path, progress_callback=None):
    """Enhanced text extraction with better error handling and progress tracking."""
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

    # Combine and clean results
    if success:
        merged_text = "\n".join([text for text in results.values() if text.strip()])
        return clean_text(merged_text)
    else:
        logger.error(f"All extraction methods failed for {file_path}")
        return ""


def parse_sections(text):
    """Parse the text into sections: Abstract, Introduction, Methodology, Results, Conclusion."""
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

    # Define section keywords and their alternatives
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
                if pos == 0 or text_lower[pos-1] in ['\n', ' ', '\t']:
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

        # Extract the section text, skipping the header
        header_end = text_lower.find('\n', start_pos)
        if header_end == -1:  # If no newline found, use a reasonable offset
            header_end = start_pos + 20

        section_text = text[header_end:end_pos].strip()
        sections[section] = section_text

    # Log if any section is empty
    for key, value in sections.items():
        if not value.strip():
            logger.warning(f"'{key}' section is empty or poorly extracted.")

    return sections


def process_pdf(file_path, progress_callback=None):
    """Process a single PDF file and return extracted data."""
    if progress_callback:
        progress_callback(f"Processing {os.path.basename(file_path)}...")

    paper_id = os.path.splitext(os.path.basename(file_path))[0]

    # Extract text
    raw_text = enhanced_extract_text_combined(file_path, progress_callback)
    if not raw_text.strip():
        logger.warning(f"No text extracted for {file_path}")
        return None

    # Parse sections
    sections = parse_sections(raw_text)
    sections = {key: clean_text(value) for key, value in sections.items()}

    # Check if all sections are empty
    if not any(sections.values()):
        logger.warning(f"No valid sections extracted for {file_path}")
        return None

    # Determine label and conference if in reference path
    label, conference = None, None
    if "reference" in file_path:
        if "non_publishable" in file_path:
            label = 0
        else:
            label = 1
            # Extract conference from path
            path_parts = file_path.split(os.sep)
            for part in path_parts:
                if part in ["cvpr", "emnlp", "kdd", "neurips", "tmlr"]:
                    conference = part.upper()
                    break

    # Create result dictionary
    result = {"Paper ID": paper_id, "Label": label, "Conference": conference}
    result.update(sections)

    return result


def process_pdfs(folder_path, output_csv, is_reference=False, progress_callback=None):
    """Process PDFs in the folder and save extracted text into a CSV."""
    data = []
    failed_pdfs = []

    # Find all PDF files
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    if progress_callback:
        progress_callback(f"Found {len(pdf_files)} PDF files to process")

    # Process PDFs with progress tracking
    for file_path in tqdm(pdf_files, desc="Processing PDFs"):
        if progress_callback:
            progress_callback(f"Processing {os.path.basename(file_path)}...")

        result = process_pdf(file_path, progress_callback)
        if result:
            data.append(result)
        else:
            failed_pdfs.append(file_path)

    # Save the results
    df = pd.DataFrame(data)
    if not df.empty:
        df.to_csv(output_csv, index=False, escapechar="\\")
        logger.info(f"Processed {len(data)} PDFs saved to {output_csv}")
        if progress_callback:
            progress_callback(f"Processed {len(data)} PDFs saved to {output_csv}")
    else:
        logger.error(f"No valid PDFs processed from {folder_path}")
        if progress_callback:
            progress_callback(f"No valid PDFs processed from {folder_path}")

    # Log failed PDFs
    if failed_pdfs:
        logger.warning(f"Failed to process {len(failed_pdfs)} PDFs: {failed_pdfs}")
        if progress_callback:
            progress_callback(f"Failed to process {len(failed_pdfs)} PDFs")

    return df


def process_pdfs_parallel(folder_path, output_csv, is_reference=False, progress_callback=None, max_workers=4):
    """Process PDFs in parallel for faster execution."""
    data = []
    failed_pdfs = []

    # Find all PDF files
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))

    if progress_callback:
        progress_callback(f"Found {len(pdf_files)} PDF files to process in parallel")

    # Process PDFs in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {executor.submit(process_pdf, pdf_file, progress_callback): pdf_file for pdf_file in pdf_files}

        for future in tqdm(concurrent.futures.as_completed(future_to_pdf), total=len(pdf_files), desc="Processing PDFs"):
            pdf_file = future_to_pdf[future]
            try:
                result = future.result()
                if result:
                    data.append(result)
                else:
                    failed_pdfs.append(pdf_file)
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                failed_pdfs.append(pdf_file)

    # Save the results
    df = pd.DataFrame(data)
    if not df.empty:
        df.to_csv(output_csv, index=False, escapechar="\\")
        logger.info(f"Processed {len(data)} PDFs saved to {output_csv}")
        if progress_callback:
            progress_callback(f"Processed {len(data)} PDFs saved to {output_csv}")
    else:
        logger.error(f"No valid PDFs processed from {folder_path}")
        if progress_callback:
            progress_callback(f"No valid PDFs processed from {folder_path}")

    # Log failed PDFs
    if failed_pdfs:
        logger.warning(f"Failed to process {len(failed_pdfs)} PDFs: {failed_pdfs}")
        if progress_callback:
            progress_callback(f"Failed to process {len(failed_pdfs)} PDFs")

    return df


if __name__ == "__main__":
    # Create processed directory if it doesn't exist
    os.makedirs("data/processed", exist_ok=True)

    # Process unlabeled papers
    process_pdfs_parallel("data/raw/papers", "data/processed/processed_papers.csv")

    # Process reference papers
    process_pdfs_parallel("data/raw/reference", "data/processed/processed_reference.csv", is_reference=True)
