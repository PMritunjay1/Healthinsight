import os
import sys
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.orchestrator import Orchestrator
from backend.ocr_module import OCRExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MedicalServer")

# FastAPI App
app = FastAPI(
    title="IEEE Multi-Agent Offline Diagnosis Server",
    description="Consolidated 100% offline local neural inference pipeline using ONNX models on CPU.",
    version="1.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow requests from React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Orchestrator instance loaded once during startup
orchestrator = None

@app.on_event("startup")
def startup_event():
    global orchestrator
    global ocr_extractor
    logger.info("FastAPI startup: Loading local agents and initializing ONNX models...")
    orchestrator = Orchestrator()
    ocr_extractor = OCRExtractor(languages=['en'])
    logger.info("FastAPI startup: All models loaded in CPU memory successfully.")

class AnalyzeRequest(BaseModel):
    patient_text: str = Field(..., description="Raw text symptoms input.")
    history: str = Field(default="", description="Patient past medical history.")

@app.post("/api/analyze")
async def analyze_case(request: AnalyzeRequest):
    global orchestrator
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Server initializing. Orchestrator not loaded yet.")
        
    logger.info(f"Received query request. Input length: {len(request.patient_text)} chars.")
    try:
        result = await orchestrator.process_case(request.patient_text, request.history)
        return result
    except Exception as e:
        logger.exception("Failed to process request through offline pipeline.")
        raise HTTPException(status_code=500, detail=f"Inference failure: {str(e)}")

from fastapi import UploadFile, File

@app.post("/api/extract_text")
async def extract_text(file: UploadFile = File(...)):
    global ocr_extractor
    if ocr_extractor is None:
        raise HTTPException(status_code=503, detail="Server initializing. OCR Extractor not loaded yet.")
        
    logger.info(f"Received document upload: {file.filename}")
    try:
        contents = await file.read()
        result = ocr_extractor.process_document(contents, file.filename)
        return result
    except Exception as e:
        logger.exception("Failed to process document upload.")
        raise HTTPException(status_code=500, detail=f"OCR failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Listen on localhost:8000
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
