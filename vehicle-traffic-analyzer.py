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
    
    def detect_device(self):
        """Detect GPU dan return optimal device"""
        if torch.cuda.is_available():
            device = 'cuda'
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU DETECTED: {gpu_name}")
            print(f"✅ CUDA Version: {torch.version.cuda}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            return device
        else:
            device = 'cpu'
            print("⚠️ GPU NOT AVAILABLE - Running on CPU")
            print("💡 Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121")
            return device
    
    def show_device_info(self):
        """Show device info popup"""
        if self.device == 'cuda':
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            message = f"🚀 GPU ACCELERATED MODE\n\n"
            message += f"Device: {gpu_name}\n"
            message += f"VRAM: {vram:.1f} GB\n\n"
            message += f"Expected Performance:\n"
            message += f"• YOLOv8l: 35-50 FPS ✅\n"
            message += f"• Processing: 5-10x faster\n"
            message += f"• Counter: Will work perfectly!"
            messagebox.showinfo("GPU Ready!", message)
        else:
            message = "⚠️ GPU NOT DETECTED\n\n"
            message += "Running on CPU mode (SLOW)\n\n"
            message += "To enable GPU:\n"
            message += "1. pip uninstall torch\n"
            message += "2. pip install torch --index-url \\\n"
            message += "   https://download.pytorch.org/whl/cu121\n"
            message += "3. Restart program"
            messagebox.showwarning("GPU Not Available", message)
