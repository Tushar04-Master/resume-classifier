import fitz  # PyMuPDF

def extract_text(file_bytes):
    # file_bytes should already be the binary content of the PDF
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    return "\n".join(page.get_text() for page in doc)