"""OCR module for rendering scanned PDF pages and extracting text via Tesseract.

Design Decision:
Instead of requiring `pdf2image` (which introduces an external `poppler-utils` dependency),
we use PyMuPDF's built-in `page.get_pixmap(dpi=300)`. PyMuPDF renders PDF pages directly in C,
converting directly to a PIL Image in memory. This reduces system dependencies down to just
`tesseract-ocr`.
"""

import logging
from PIL import Image
import fitz  # PyMuPDF
import pytesseract

logger = logging.getLogger(__name__)


def ocr_pdf_page(page: fitz.Page, dpi: int = 300) -> str:
    """Render a PyMuPDF Page into an in-memory image and run Tesseract OCR.

    Args:
        page: PyMuPDF Page object to be rendered.
        dpi: Resolution for rendering. 300 DPI is the sweet spot for OCR accuracy
             on fine insurance contract typography without excessive memory usage.

    Returns:
        Extracted text string from the scanned image.
    """
    try:
        # Render page to RGB pixel map in memory at 300 DPI
        pix = page.get_pixmap(dpi=dpi)

        # Convert raw pixel bytes directly to PIL Image (no temporary files on disk)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # Run Tesseract OCR on the image
        text = pytesseract.image_to_string(img)
        return text.strip()

    except pytesseract.TesseractNotFoundError:
        logger.warning(
            f"Tesseract is not installed or not in PATH. Could not OCR page {page.number + 1}. "
            "Install 'tesseract-ocr' or run 'apt install tesseract-ocr' on Linux."
        )
        return ""
    except Exception as exc:
        logger.error(f"OCR failed on page {page.number + 1}: {exc}")
        return ""
