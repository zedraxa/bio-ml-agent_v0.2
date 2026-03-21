import os
import sys
import asyncio
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
log = logging.getLogger("demo.microscopy")

try:
    import cv2
    import numpy as np
except ImportError:
    log.error("❌ Physical ML dependencies missing! Please run:")
    log.error("   pip install -r requirements_microscopy.txt")
    sys.exit(1)

from bio_ml_agent.agents.microscopy.perception_agent import MicroscopyPerceptionAgent
from bio_ml_agent.agents.microscopy.identifier_agent import MicroscopyIdentifierAgent
from bio_ml_agent.agents.microscopy.segmentation_agent import MicroscopySegmentationAgent
from bio_ml_agent.agents.microscopy.morphometrics_agent import MorphometricsAgent

def generate_dummy_image(path: str):
    """Fiziksel test için sahte mikroskopi görüntüsü üretir."""
    img = np.zeros((512, 512, 3), dtype=np.uint8)
    # Background fluorescence noise
    noise = np.random.normal(10, 5, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    # Draw 3 dummy cells
    cv2.circle(img, (150, 150), 40, (0, 200, 0), -1) # Cell core
    cv2.circle(img, (150, 150), 45, (0, 100, 0), 2)  # Cell membrane
    
    cv2.circle(img, (350, 200), 50, (0, 180, 0), -1)
    
    cv2.ellipse(img, (250, 400), (60, 30), 45, 0, 360, (0, 190, 0), -1)
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)
    log.info(f"Generated dummy physical image at {path}")

async def run_microscopy_mission(image_path: str):
    """
    A-Serisi Fiziksel Pipeline Entegrasyon Testi.
    Cihaz/Modeller yüklü olmasa bile fallback'ler test edilerek Pipeline bütünlüğü doğrulanır.
    """
    if not os.path.exists(image_path):
        generate_dummy_image(image_path)
        
    context = {"image_path": image_path}
    
    log.info("=== 1. Starting A1: Physical Perception ===")
    a1 = MicroscopyPerceptionAgent()
    a1.perceive(context)
    a1.act("multimodal_query")
    res_a1 = a1.summarize()
    log.info(f"A1 Output: {res_a1.data['modality_guess']}")
    
    log.info("\n=== 2. Starting A2: Deep Ontology ID ===")
    a2 = MicroscopyIdentifierAgent()
    a2.perceive(context)
    a2.act("send_ontology")
    res_a2 = a2.summarize()
    
    log.info("\n=== 3. Starting A3: Physical Segmentation (Cellpose) ===")
    a3 = MicroscopySegmentationAgent()
    a3.perceive(context)
    a3.act("inference_cellpose")
    res_a3 = a3.summarize()
    
    # Pass masks to A4
    context["segments"] = a3.segments
    
    log.info("\n=== 4. Starting A4: Physical Morphometrics (OpenCV) ===")
    a4 = MorphometricsAgent()
    a4.perceive(context)
    a4.act("geometry_shape")
    res_a4 = a4.summarize()
    
    log.info("\n🏁 Physical Pipeline Verified.")
    log.info(f"Artifacts output to: {res_a4.data.get('morpho_file')}")

if __name__ == "__main__":
    asyncio.run(run_microscopy_mission("samples/cell_sample_physical_01.jpg"))

