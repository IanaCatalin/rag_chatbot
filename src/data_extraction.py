import json
import glob
import PyPDF2
from typing import List, Dict, Union


def extract_text_from_pdf(pdf_path: Union[str, List[str]]) -> str:
    """
    Extract text content from one or multiple PDF files.

    Args:
        pdf_path: Path to a PDF file or a glob pattern to match multiple PDF files

    Returns:
        A string containing the extracted text from all PDFs
    """
    all_text = ""

    # Handle glob patterns
    if isinstance(pdf_path, str) and ('*' in pdf_path or '?' in pdf_path):
        pdf_files = glob.glob(pdf_path)
    elif isinstance(pdf_path, str):
        pdf_files = [pdf_path]
    else:
        pdf_files = pdf_path

    for pdf_file in pdf_files:
        try:
            with open(pdf_file, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        all_text += page_text + "\n\n"
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

    return all_text


def extract_from_json(json_path: str, include_urls: bool = True) -> Dict[str, str]:
    """
    Extract text from a JSON file containing URLs and text.

    Args:
        json_path: Path to the JSON file
        include_urls: Whether to include the URLs at the top of each text content (default: True)

    Returns:
        A dictionary mapping URLs to their corresponding text content
    """
    result = {}
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    url = item.get('url', f'item_{i}')
                    text = item.get('text', '')
                    # Format the content with or without URL based on the parameter
                    if include_urls:
                        result[url] = f"{url}\n\n{text}\n\n\n\n"
                    else:
                        result[url] = f"{text}\n\n\n\n"
    except Exception as e:
        print(f"Error processing JSON file: {str(e)}")

    return result
