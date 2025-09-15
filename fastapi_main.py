from src.config import Config, init_environment
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Revo Insurance Analytics API",
    description="API for generating PDF reports from financial analysis results",
    version="1.0.0",
    docs_url="/docs"
)

@app.on_event("startup")
async def startup_event():
    logger.info("API Server started")
    logger.info("API Documentation available at: http://localhost:8000/docs")
    logger.info("Health check endpoint: http://localhost:8000/health")
    logger.info("Analysis endpoint: http://localhost:8000/run-analysis")

@app.get("/run-analysis", 
    summary="Generate Analysis PDF",
    description="Creates a PDF report containing all analysis visualizations",
    response_description="PDF file containing analysis figures"
)


async def run_analysis():

    cfg = Config()
    paths = init_environment(cfg)

    # 2. Get output images
    figures_path = Path(paths["FIG_DIR"])
    images = sorted(figures_path.glob("*.png"))
    
    if not images:
        raise HTTPException(status_code=404, detail="No output images found")
    
    # 3. Generate PDF
    pdf_path = Path("temp/analysis_report.pdf")
    pdf_path.parent.mkdir(exist_ok=True)
    
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    # Add each image to PDF
    for i, img_path in enumerate(images):
        if i > 0:  # New page for each image after the first
            c.showPage()
        c.drawImage(str(img_path), 50, height - 550, width=500, height=500)
    
    c.save()
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="analysis_report.pdf"
    )
        

@app.get("/health",
    summary="Health Check",
    description="Check if the API is running properly",
    response_description="Returns health status"
)
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)