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
