from fastapi import FastAPI, UploadFile, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize FastAPI app
app = FastAPI()

# Enable CORS (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with specific origins in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model (use pre-trained weights)
model = YOLO("yolo11n.pt")  # YOLOv8n: Smallest model for performance

@app.get("/")
async def root():
    return {"message": "YOLO FastAPI is running"}

@app.post("/process_frame")
async def process_frame(file: UploadFile):
    """
    Endpoint to process a single image frame and return detections.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    # Read and decode the uploaded image
    image_bytes = await file.read()
    np_image = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

    # Resize frame for faster YOLO inference
    frame = cv2.resize(frame, (640, 640))

    # Perform YOLO object detection
    results = model.predict(frame)

    # Extract detections
    detections = []
    for r in results[0].boxes:
        bbox = r.xyxy.tolist()[0]  # Bounding box [x1, y1, x2, y2]
        conf = r.conf.tolist()[0]  # Confidence score
        cls = int(r.cls.tolist()[0])  # Class index
        class_label = model.names[cls]  # Get class label from YOLO model
        detections.append({"bbox": bbox, "confidence": conf, "class": class_label})

    return {"detections": detections}

@app.websocket("/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time object detection.
    """
    await websocket.accept()
    try:
        while True:
            # Receive frame as bytes from the client
            data = await websocket.receive_bytes()
            np_image = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

            # Resize frame for faster YOLO inference
            frame = cv2.resize(frame, (640, 640))

            # Perform YOLO object detection
            results = model.predict(frame)

            # Extract detections
            detections = []
            for r in results[0].boxes:
                bbox = r.xyxy.tolist()[0]  # Bounding box [x1, y1, x2, y2]
                conf = r.conf.tolist()[0]  # Confidence score
                cls = int(r.cls.tolist()[0])  # Class index
                class_label = model.names[cls]  # Get class label from YOLO model
                detections.append({"bbox": bbox, "confidence": conf, "class": class_label})

            # Send detections to the client as JSON
            await websocket.send_json({"detections": detections})

    except Exception as e:
        print(f"WebSocket Error: {e}")
        await websocket.close()

