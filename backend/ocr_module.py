import os
import io
import logging
import cv2
import numpy as np
import easyocr
import fitz  # PyMuPDF
from typing import Dict, Any

logger = logging.getLogger("OCR_Module")

class OCRExtractor:
    def __init__(self, languages=['en']):
        logger.info("Initializing EasyOCR reader (CPU)...")
        # Initialize easyocr, using CPU by default for broader compatibility
        self.reader = easyocr.Reader(languages, gpu=False)
        
    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Applies basic preprocessing for OCR."""
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image bytes.")
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Denoising
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        
        return denoised

    def _extract_from_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """Extracts text from a single image."""
        try:
            preprocessed_img = self._preprocess_image(image_bytes)
            # EasyOCR returns list of (bbox, text, confidence)
            results = self.reader.readtext(preprocessed_img, paragraph=False)
            
            if not results:
                return {
                    "source_type": "image",
                    "ocr_text": "",
                    "ocr_confidence": 0.0,
                    "processing_status": "warning",
                    "warnings": ["No readable clinical text was detected in the image."]
                }
            
            extracted_text = []
            confidences = []
            
            for (bbox, text, conf) in results:
                extracted_text.append(text)
                confidences.append(conf)
                
            avg_conf = sum(confidences) / len(confidences)
            
            return {
                "source_type": "image",
                "ocr_text": "\n".join(extracted_text),
                "ocr_confidence": round(avg_conf, 2),
                "processing_status": "success",
                "warnings": []
            }
            
        except Exception as e:
            logger.error(f"Image OCR failed: {str(e)}")
            return {
                "source_type": "image",
                "ocr_text": "",
                "ocr_confidence": 0.0,
                "processing_status": "error",
                "warnings": [f"Image processing failed: {str(e)}"]
            }

    def _extract_from_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extracts text from a PDF, falling back to OCR if scanned."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = []
            is_scanned = True
            
            # Check if PDF contains selectable text
            for page in doc:
                text = page.get_text("text").strip()
                if len(text) > 50:  # Arbitrary threshold to decide if it's text-based
                    is_scanned = False
                    full_text.append(text)
                    
            if not is_scanned:
                return {
                    "source_type": "pdf_text",
                    "ocr_text": "\n".join(full_text),
                    "ocr_confidence": 1.0,
                    "processing_status": "success",
                    "warnings": []
                }
                
            # If scanned, render pages to images and run OCR
            logger.info("PDF appears to be scanned. Running OCR on pages...")
            ocr_texts = []
            confidences = []
            warnings = []
            
            for page in doc:
                # Render page to an image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                img_bytes = pix.tobytes("png")
                
                page_result = self._extract_from_image(img_bytes)
                if page_result["processing_status"] == "success":
                    ocr_texts.append(page_result["ocr_text"])
                    confidences.append(page_result["ocr_confidence"])
                else:
                    warnings.extend(page_result["warnings"])
                    
            if not ocr_texts:
                return {
                    "source_type": "pdf_scan",
                    "ocr_text": "",
                    "ocr_confidence": 0.0,
                    "processing_status": "warning",
                    "warnings": ["No readable clinical text was detected in the PDF scans."] + warnings
                }
                
            avg_conf = sum(confidences) / len(confidences)
            
            return {
                "source_type": "pdf_scan",
                "ocr_text": "\n".join(ocr_texts),
                "ocr_confidence": round(avg_conf, 2),
                "processing_status": "success",
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"PDF OCR failed: {str(e)}")
            return {
                "source_type": "pdf",
                "ocr_text": "",
                "ocr_confidence": 0.0,
                "processing_status": "error",
                "warnings": [f"PDF processing failed: {str(e)}"]
            }

    def process_document(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        """Main entry point for processing an uploaded document."""
        ext = filename.split('.')[-1].lower() if '.' in filename else ''
        
        logger.info(f"Processing document: {filename}")
        
        if ext == 'pdf':
            return self._extract_from_pdf(file_bytes)
        elif ext in ['jpg', 'jpeg', 'png']:
            return self._extract_from_image(file_bytes)
        else:
            return {
                "source_type": "unknown",
                "ocr_text": "",
                "ocr_confidence": 0.0,
                "processing_status": "error",
                "warnings": [f"Unsupported file format: {ext}"]
            }
