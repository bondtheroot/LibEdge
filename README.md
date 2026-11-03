# LibEdge // Multi-Camera Dynamic Seat Topology Manager

LibEdge is a real-time computer vision and IoT management system designed to monitor library seat occupancy across different floors and zones using YOLOv11 vision models, FastAPI, and MongoDB.

## Features

- **AI-Powered Occupancy Detection:** Uses YOLOv11 pose estimation and object detection to identify chairs and people in sitting postures.
- **Dynamic Association:** Associates sitting individuals with chairs using bounding box intersection (IoU) and keypoint analysis.
- **MongoDB Backend:** Persists camera configurations, streams, and latest occupancy metrics via MongoDB URIs.
- **Background Polling Scheduler:** Automatically samples camera video feeds at configured intervals (e.g., 15s, 30s, 60s).
- **Modern Web Dashboard:** Real-time metrics summary, camera CRUD operations, and live test triggers.

## Prerequisites

- Python 3.10+
- MongoDB instance (local or cloud-hosted via connection URI, e.g., `mongodb+srv://user:pass@cluster.mongodb.net/`)
- PyTorch and Ultralytics YOLO dependencies
- OpenCV

## Installation

1. Clone the repository and navigate to the project directory.
2. Install the required Python dependencies:
   ```bash
   pip install fastapi uvicorn pymongo opencv-python ultralytics torch jinja2 pydantic
   ```
3. Configure your MongoDB connection URI (optional, defaults to `mongodb://localhost:27017/`):
   ```bash
   export MONGO_URL="mongodb://localhost:27017/"
   ```

## Running the Application

Start the FastAPI application using Uvicorn:

