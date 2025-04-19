import fitz  # PyMuPDF
import io
import re
import base64
import hashlib
from PIL import Image
from pydantic import BaseModel, Field

from langchain_core.documents import Document


def pdf_page_to_base64(pdf_path: str) -> str:
    """
    Converts the first page of a PDF file to a base64-encoded PNG image.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Base64-encoded string of the first page as a PNG image.
    """
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(0)  # Load the first page
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def make_hash_from_metadata(metadata: dict[str, str]):
    # Convert the dictionary to a sorted string representation
    sorted_data = str(sorted(metadata.items()))

    # Create a hash using SHA-256
    hash_object = hashlib.sha256(sorted_data.encode())
    hash_value = hash_object.hexdigest()
    return hash_value[:8]

def extract_titles_and_sections(pages: list[Document], metadata_extra: dict[str, str], meta_hash: str) -> list[dict[str, str]]:
    """
    Extracts titles and sections from the PDF pages.
    Args:
        pages (list[Document]): List of Document objects representing the PDF pages.
        metadata_extra (dict[str, str]): Additional metadata to include in the output.
    Returns:
        list[dict[str, str]]: List of dictionaries containing titles, metadata, and content.
    """
    title_pattern = r'^(?:\d+\.?\s*|\bI\b\s*)?[A-Z][A-Za-z\-:]*\s*(?:[Aa][Nn][Dd]\s+[A-Z][A-Za-z\-:]*\s*)?$'  # tweak as needed
    metadata = {i: j for i, j in pages[0].metadata.items() if i not in ["page_label", "page"]} # remove source and page from metadata
    metadata["hash"] = meta_hash
    metadata.update(metadata_extra)
    sections = []
    current_title = None
    current_content = []

    for page in pages:
        lines = page.page_content.split('\n')
        for line in lines:
            if re.match(title_pattern, line.strip()):
                if current_title:
                    sections.append({
                        'title': current_title,
                        'metadata': metadata,
                        'content': "\n".join(current_content)
                    })
                current_title = line.strip()
                current_content = []
            else:
                current_content.append(line.strip())

    # Add the last section
    if current_title:
        sections.append({
            'title': current_title,
            'metadata': metadata,
            'content': "\n".join(current_content)
        })

    return sections


def preprocess_pdf(docs: list[Document], paper_meta: dict[str, str], meta_hash: str) -> list[dict[str, str]]:
    """
    Preprocesses the PDF documents by extracting titles and sections, filtering out unwanted sections,
    and cleaning the content.
    Args:
        docs (list[Document]): List of Document objects representing the PDF pages.
        paper_meta (dict[str, str]): Metadata of the paper.
    Returns:
        list[dict[str, str]]: List of dictionaries containing titles, metadata, and cleaned content.
    """
    sections = extract_titles_and_sections(docs, paper_meta, meta_hash)

    # Filter based on keywords
    filtered = []
    for s in sections:
        if all(k not in s["title"].lower() for k in ["references", "introduction"]):
            filtered.append(s)
        elif s["title"].lower() == "introduction":
            intro = s["content"].split('.\n')
            store = [i for i in intro if not re.search(r'\[\d+\]', i)]
            s["content"] = "\n".join(store)
            filtered.append(s)

    return filtered
