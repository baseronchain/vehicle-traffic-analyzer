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
