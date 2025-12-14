import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import torch

class TrafficDetectorGPU:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Detector - GPU ACCELERATED 🚀")
        self.root.geometry("1200x750")
        
        # GPU Detection
        self.device = self.detect_device()
        self.show_device_info()
        
        # Load model dengan GPU
        self.load_model()
        
        # Vehicle classes
        self.vehicle_classes = {
            2: 'mobil',
            3: 'motor',
            5: 'bus',
            7: 'truck'
        }
        
        # Tracking
        self.tracked_vehicles = {}
        self.counted_ids = set()
        
        # Settings - OPTIMIZED untuk GPU
        self.counting_line_y = 280
        self.line_offset = 40  # Good balance
        self.confidence_threshold = 0.5  # Optimal dengan GPU
        
        # Counters
        self.reset_counters()
        
        # Video
        self.cap = None
        self.is_running = False
        self.video_thread = None
        self.frame_width = 640
        self.frame_height = 480
        
        # Performance
        self.total_frames = 0
        self.detection_count = 0
        
        self.setup_gui()
