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
    def load_model(self):
        """Load YOLO model dengan device specification"""
        model_name = 'yolov8l.pt' if self.device == 'cuda' else 'yolov8n.pt'
        
        try:
            print(f"Loading {model_name} on {self.device}...")
            self.model = YOLO(model_name)
            
            # Force model ke device
            self.model.to(self.device)
            
            print(f"✅ Model loaded on {self.device}")
            
            if self.device == 'cuda':
                print(f"✅ Model: YOLOv8l (High Accuracy)")
                print(f"✅ Expected FPS: 35-50")
            else:
                print(f"⚠️ Model: YOLOv8n (fallback for CPU)")
                print(f"⚠️ Expected FPS: 8-12")
                
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            messagebox.showerror("Error", f"Failed to load model: {e}")

def reset_counters(self):
    self.vehicle_counts = {
        'mobil': 0,
        'motor': 0,
        'bus': 0,
        'truck': 0
    }
    self.total_vehicles = 0
    self.tracked_vehicles = {}
    self.counted_ids = set()
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
