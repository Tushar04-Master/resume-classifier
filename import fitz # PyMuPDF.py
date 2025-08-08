import fitz  # PyMuPDF

def extract_text(pdf_input):
    # Accept bytes, bytearray, or file-like object
    if hasattr(pdf_input, "read"):
        data = pdf_input.read()
    elif isinstance(pdf_input, (bytes, bytearray)):
        data = pdf_input
    else:
        raise ValueError("Input must be bytes, bytearray, or file-like object.")

    doc = fitz.open(stream=data, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text
