import os
import unittest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pymupdf as fitz
import io
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.ocr_module import OCRExtractor

class TestOCRExtractor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ocr = OCRExtractor(languages=['en'])
        cls.test_dir = os.path.join(os.path.dirname(__file__), 'test_assets')
        os.makedirs(cls.test_dir, exist_ok=True)
        
    def _create_test_image(self, text, filename, noise=False):
        img = Image.new('RGB', (800, 400), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        # Without specifying a TTF font, the default font is very small. We'll just use simple text.
        d.text((10,10), text, fill=(0,0,0))
        
        if noise:
            # Add some light noise for low-quality test
            noise_arr = np.random.randint(0, 20, (400, 800, 3), dtype=np.uint8)
            img_arr = np.array(img)
            noisy_img = np.clip(img_arr + noise_arr, 0, 255).astype(np.uint8)
            img = Image.fromarray(noisy_img)
            
        filepath = os.path.join(self.test_dir, filename)
        img.save(filepath)
        return filepath

    def _create_test_pdf(self, text, filename, is_image_based=False):
        filepath = os.path.join(self.test_dir, filename)
        if not is_image_based:
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text(fitz.Point(50, 50), text)
            doc.save(filepath)
            doc.close()
        else:
            img_path = self._create_test_image(text, filename.replace('.pdf', '.png'))
            doc = fitz.open()
            page = doc.new_page()
            page.insert_image(page.rect, filename=img_path)
            doc.save(filepath)
            doc.close()
        return filepath

    def test_clear_typed_report(self):
        text = "Patient: John\nAge: 45\nChief Complaint: Severe chest pain."
        filepath = self._create_test_image(text, "clear_report.jpg")
        with open(filepath, "rb") as f:
            res = self.ocr.process_document(f.read(), "clear_report.jpg")
        self.assertEqual(res["processing_status"], "success")
        self.assertIn("John", res["ocr_text"])

    def test_low_quality_scanned_report(self):
        text = "Patient History: Hypertension and type 2 diabetes. The patient is doing well."
        filepath = self._create_test_image(text, "noisy_report.jpg", noise=True)
        with open(filepath, "rb") as f:
            res = self.ocr.process_document(f.read(), "noisy_report.jpg")
        self.assertIn(res["processing_status"], ["success", "warning"])

    def test_pdf_selectable_text(self):
        text = "Laboratory Results:\nGlucose: 120 mg/dL\nHR: 80 bpm\nThis is enough text to pass the length threshold of fifty characters for sure."
        filepath = self._create_test_pdf(text, "selectable.pdf", is_image_based=False)
        with open(filepath, "rb") as f:
            res = self.ocr.process_document(f.read(), "selectable.pdf")
        self.assertEqual(res["source_type"], "pdf_text")
        self.assertIn("Glucose", res["ocr_text"])

    def test_pdf_scanned_pages(self):
        text = "Assessment: Patient is stable. Continue medication."
        filepath = self._create_test_pdf(text, "scanned.pdf", is_image_based=True)
        with open(filepath, "rb") as f:
            res = self.ocr.process_document(f.read(), "scanned.pdf")
        self.assertEqual(res["source_type"], "pdf_scan")
        self.assertIn("stable", res["ocr_text"].lower())

    def test_no_readable_text(self):
        filepath = self._create_test_image("", "empty.jpg")
        with open(filepath, "rb") as f:
            res = self.ocr.process_document(f.read(), "empty.jpg")
        self.assertEqual(res["processing_status"], "warning")
        self.assertIn("No readable", res["warnings"][0])

if __name__ == '__main__':
    unittest.main()
