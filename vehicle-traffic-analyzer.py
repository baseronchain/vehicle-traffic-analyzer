import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import torch
from queue import Queue, Empty
import time

class TrafficDetectorGPU:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistem pngdeteksi, penghitung, dan klasifikasi kendaraan dengan YOLOv8l")
        self.root.geometry("1200x750")
        
        # GPU Detectsion
        self.device = self.detect_device()
        self.show_device_info()
        
        # Loadd model dengan GPU
        self.load_model()
        
        # kals kendaraan
        self.vehicle_classes = {
            2: 'mobil',
            3: 'motor',
            5: 'bus',
            7: 'truck'
        }
         # untuk tracking
 self.tracked_vehicles = {}
 self.counted_ids = set()
 
 
 self.counting_line_y = 280
 self.line_offset = 40  # balancee
 self.confidence_threshold = 0.5  
 
 
 self.reset_counters()
 
 
 self.cap = None
 self.is_running = False
 self.video_thread = None
 self.frame_width = 640
 self.frame_height = 480
 
 
 self.total_frames = 0
 self.detection_count = 0
 
 self.setup_gui()
    # GUI rendering throttling (Tkinter is a common FPS bottleneck)
    self.target_display_fps = 20  # render rate, not detection rate
    self._gui_interval_ms = max(1, int(1000 / self.target_display_fps))
    self.frame_queue = Queue(maxsize=1)  # keep only latest frame
    self._track_call_mode = 0  # 0=unknown, 1=half+imgsz ok, 2=imgsz ok, 3=basic only
    self.infer_imgsz = 640
    self.use_half = (self.device == 'cuda')
    self.root.after(self._gui_interval_ms, self.gui_update_loop)

def detect_device(self):
    """Sistem pngdeteksi, penghitung, dan klasifikasi kendaraan dengan YOLOv8l"""
    if torch.cuda.is_available():
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU DETECTED: {gpu_name}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        return device
    else:
        device = 'cpu'
        print("")
        print("")
        return device

def show_device_info(self):
    """Show device info popup"""
    if self.device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        message = f"Sistem pngdeteksi, penghitung, dan klasifikasi kendaraan dengan YOLOv8l\n\n"
        message += f"Device: {gpu_name}\n"
        message += f"VRAM: {vram:.1f} GB\n\n"
        message += f""
        message += f""
        message += f""
        message += f""
        messagebox.showinfo("GPU Ready!", message)
    else:
        message = "GPU NOT DETECTED\n\n"
        message += "Running on CPU mode (SLOW)\n\n"
        message += "To enable GPU:\n"
        message += "1. pip uninstall torch\n"
        message += "2. pip install torch --index-url \\\n"
        message += "   https://download.pytorch.org/whl/cu121\n"
        message += "3. Restart program"
        messagebox.showwarning("GPU Not Available", message)

def load_model(self):
    """"""
    model_name = 'yolov8l.pt' if self.device == 'cuda' else 'yolov8n.pt'
    
    try:
        print(f"Loading {model_name} on {self.device}...")
        self.model = YOLO(model_name)
            
