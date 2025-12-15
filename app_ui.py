"""Tkinter UI for interactive image compression experiments (grid, quadtree, k-d tree)."""

# COMP6049001 - Algorithm Design and Analysis
# Final Project: A Comparative Analysis of Image Compression Algorithms
# Group Members:
# - Harris Ekaputra Suryadi (2802400502)
# - Michael Arianno Chandrarieta (2802499711)
# - Muhammad Ryan Ismail Putra (2802522733)

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageEnhance
import time
import io
import threading
import os

from algorithms import compress_kdtree, compress_quadtree, compress_uniform_grid
from image_analyzer import ImageAnalyzer

# --- Colors & Styles ---
COLOR_BG = "#ecf0f1"
COLOR_SIDEBAR = "#2c3e50"
COLOR_TEXT_WHITE = "#ffffff"
COLOR_TEXT_DARK = "#2c3e50"
COLOR_ACCENT = "#3498db"
COLOR_ACCENT_HOVER = "#2980b9"
COLOR_BAR_ORIGINAL = "#e74c3c"
COLOR_BAR_COMPRESSED = "#2ecc71"
COLOR_CROP_OUTLINE = "#e74c3c"


class ImageCompressorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Compression Workbench")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLOR_BG)
        
        # Image State
        self.original_filepath = None
        self.original_filesize = 0
        
        # We maintain three image states:
        # 1. self.raw_image: The original image loaded from disk (full size).
        # 2. self.base_working_image: The image after CROP/ROTATE is applied (but before filters).
        # 3. self.current_image: The image after CROP + FILTERS are applied. This is what gets compressed.
        self.raw_image = None 
        self.base_working_image = None 
        self.current_image = None
        
        self.display_image_obj = None
        self.image_tk = None
        self.original_tk = None 
        self.slider_var = tk.StringVar(value="25")
        
        # Crop State
        self.is_cropping = False
        self.crop_start = None
        self.crop_rect_id = None
        
        # Filter State
        self.grayscale_var = tk.BooleanVar(value=False)
        self.brightness_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)
        self.show_grid_var = tk.BooleanVar(value=False)
        self.last_blocks = None

        # UI Styles
        self.font_header = ("Segoe UI", 12, "bold")
        self.font_normal = ("Segoe UI", 10)
        
        self.setup_ui()
        self.set_controls_state('disabled')

    def set_controls_state(self, state):
        # Toggle all actionable controls at once (used on load/processing states).
        for widget in self.algo_frame.winfo_children(): widget.config(state=state)
        self.slider.config(state=state)
        self.save_button.config(state=state)
        self.slider_entry.config(state=state)
        self.compress_button.config(state=state)
        self.compare_button.config(state=state)
        self.crop_button.config(state=state)
        
        # Filter controls
        self.chk_grayscale.config(state=state)
        self.scale_brightness.config(state=state)
        self.scale_contrast.config(state=state)
        self.btn_reset_edits.config(state=state)
        self.chk_grid.config(state=state)
        
        # Transform controls
        self.btn_rot_left.config(state=state)
        self.btn_rot_right.config(state=state)
        self.btn_flip_h.config(state=state)
        self.btn_flip_v.config(state=state)

    def setup_ui(self):
        # --- Sidebar (Left) ---
        sidebar = tk.Frame(self.root, bg=COLOR_SIDEBAR, width=320, padx=15, pady=15)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="COMPRESSION\nANALYZER", font=("Segoe UI", 18, "bold"), 
                 bg=COLOR_SIDEBAR, fg=COLOR_TEXT_WHITE, justify=tk.LEFT).pack(anchor="w", pady=(0, 20))

        # Notebook for Tabs
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background=COLOR_SIDEBAR, borderwidth=0)
        style.configure("TNotebook.Tab", background="#34495e", foreground="white", padding=[10, 5], font=self.font_normal)
        style.map("TNotebook.Tab", background=[("selected", COLOR_BG)], foreground=[("selected", "black")])
        
        self.notebook = ttk.Notebook(sidebar)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=5)

        # --- TAB 1: EDIT (Crop & Filters) ---
        self.tab_edit = tk.Frame(self.notebook, bg=COLOR_SIDEBAR, padx=5, pady=10)
        self.notebook.add(self.tab_edit, text="Editor")

        # File Operations
        self.load_button = tk.Button(self.tab_edit, text="📂 Load Image", font=self.font_normal, 
                                     bg=COLOR_ACCENT, fg="white", relief=tk.FLAT, padx=10, pady=5, command=self.load_image)
        self.load_button.pack(fill=tk.X, pady=5)

        # Crop Section
        tk.Label(self.tab_edit, text="Crop & Resize", font=self.font_header, bg=COLOR_SIDEBAR, fg=COLOR_TEXT_WHITE).pack(anchor="w", pady=(15, 5))
        self.crop_button = tk.Button(self.tab_edit, text="✂ Start Crop Mode", font=self.font_normal,
                                     bg="#e67e22", fg="white", relief=tk.FLAT, padx=10, pady=5, command=self.start_crop_mode)
        self.crop_button.pack(fill=tk.X, pady=5)

        # Rotate & Flip Section
        tk.Label(self.tab_edit, text="Rotate & Flip", font=self.font_header, bg=COLOR_SIDEBAR, fg=COLOR_TEXT_WHITE).pack(anchor="w", pady=(15, 5))
        
        # Use Grid layout for perfect alignment
        transform_frame = tk.Frame(self.tab_edit, bg=COLOR_SIDEBAR)
        transform_frame.pack(fill=tk.X)
        transform_frame.columnconfigure(0, weight=1)
        transform_frame.columnconfigure(1, weight=1)

        self.btn_rot_left = tk.Button(transform_frame, text="↺ 90°", bg="#7f8c8d", fg="white", relief=tk.FLAT, command=lambda: self.apply_transform("rotate_left"))
        self.btn_rot_left.grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        
        self.btn_rot_right = tk.Button(transform_frame, text="↻ 90°", bg="#7f8c8d", fg="white", relief=tk.FLAT, command=lambda: self.apply_transform("rotate_right"))
        self.btn_rot_right.grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        
        self.btn_flip_h = tk.Button(transform_frame, text="↔ Flip H", bg="#7f8c8d", fg="white", relief=tk.FLAT, command=lambda: self.apply_transform("flip_h"))
        self.btn_flip_h.grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        
        self.btn_flip_v = tk.Button(transform_frame, text="↕ Flip V", bg="#7f8c8d", fg="white", relief=tk.FLAT, command=lambda: self.apply_transform("flip_v"))
        self.btn_flip_v.grid(row=1, column=1, sticky="ew", padx=2, pady=2)

        # Filters Section
        tk.Label(self.tab_edit, text="Adjustments", font=self.font_header, bg=COLOR_SIDEBAR, fg=COLOR_TEXT_WHITE).pack(anchor="w", pady=(20, 5))
        
        self.chk_grayscale = tk.Checkbutton(self.tab_edit, text="Grayscale Mode", variable=self.grayscale_var, 
                                            bg=COLOR_SIDEBAR, fg="white", selectcolor=COLOR_SIDEBAR, activebackground=COLOR_SIDEBAR, 
                                            font=self.font_normal, command=self.apply_edits)
        self.chk_grayscale.pack(anchor="w", pady=2)
        
        tk.Label(self.tab_edit, text="Brightness", bg=COLOR_SIDEBAR, fg="#bdc3c7").pack(anchor="w", pady=(5,0))
        self.scale_brightness = tk.Scale(self.tab_edit, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                         variable=self.brightness_var, bg=COLOR_SIDEBAR, fg="white", highlightthickness=0, troughcolor="#34495e", command=lambda x: self.apply_edits())
        self.scale_brightness.pack(fill=tk.X)

        tk.Label(self.tab_edit, text="Contrast", bg=COLOR_SIDEBAR, fg="#bdc3c7").pack(anchor="w", pady=(5,0))
        self.scale_contrast = tk.Scale(self.tab_edit, from_=0.5, to=2.0, resolution=0.1, orient=tk.HORIZONTAL, 
                                       variable=self.contrast_var, bg=COLOR_SIDEBAR, fg="white", highlightthickness=0, troughcolor="#34495e", command=lambda x: self.apply_edits())
        self.scale_contrast.pack(fill=tk.X)

        self.btn_reset_edits = tk.Button(self.tab_edit, text="↺ Reset All Edits", bg="#e74c3c", fg="white", relief=tk.FLAT, command=self.reset_edits)
        self.btn_reset_edits.pack(fill=tk.X, pady=(20, 0))


        # --- TAB 2: COMPRESS (Algorithms & Stats) ---
        self.tab_compress = tk.Frame(self.notebook, bg=COLOR_SIDEBAR, padx=5, pady=10)
        self.notebook.add(self.tab_compress, text="Compress")

        # Algorithm Selection
        tk.Label(self.tab_compress, text="Algorithm", font=self.font_header, bg=COLOR_SIDEBAR, fg=COLOR_TEXT_WHITE).pack(anchor="w", pady=(0, 5))
        self.algo_frame = tk.Frame(self.tab_compress, bg=COLOR_SIDEBAR)
        self.algo_frame.pack(fill=tk.X)
        
        self.selected_algo = tk.StringVar(value="Quadtree")
        
        for text, value in [("Uniform Grid", "Uniform Grid"), ("Quadtree", "Quadtree"), ("k-d Tree", "k-d Tree")]:
            rb = tk.Radiobutton(self.algo_frame, text=text, variable=self.selected_algo, value=value, 
                                bg=COLOR_SIDEBAR, fg="white", selectcolor=COLOR_SIDEBAR, activebackground=COLOR_SIDEBAR, 
                                activeforeground="white", font=self.font_normal, command=self.run_compression)
            rb.pack(anchor="w", pady=2)

        # Slider Controls
        tk.Label(self.tab_compress, text="Threshold / Block Size", font=self.font_header, bg=COLOR_SIDEBAR, fg=COLOR_TEXT_WHITE).pack(anchor="w", pady=(15, 5))
        
        slider_frame = tk.Frame(self.tab_compress, bg=COLOR_SIDEBAR)
        slider_frame.pack(fill=tk.X)
        
        self.slider = tk.Scale(slider_frame, from_=1, to=100, orient=tk.HORIZONTAL, variable=self.slider_var, 
                               showvalue=0, bg=COLOR_SIDEBAR, fg="white", highlightthickness=0, troughcolor="#34495e")
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        validate_cmd = (self.root.register(self.validate_input), '%P')
        self.slider_entry = tk.Entry(slider_frame, textvariable=self.slider_var, width=4, font=self.font_normal, 
                                     justify="center", validate='key', validatecommand=validate_cmd)
        self.slider_entry.pack(side=tk.LEFT, padx=(10, 0))

        self.slider_info_label = tk.Label(self.tab_compress, text="Value: 25", font=("Segoe UI", 9), bg=COLOR_SIDEBAR, fg="#bdc3c7")
        self.slider_info_label.pack(anchor="w")

        # Grid Toggle
        self.chk_grid = tk.Checkbutton(self.tab_compress, text="Draw Grid Lines", variable=self.show_grid_var, 
                                       bg=COLOR_SIDEBAR, fg="white", selectcolor=COLOR_SIDEBAR, activebackground=COLOR_SIDEBAR, 
                                       font=self.font_normal, command=self.refresh_display)
        self.chk_grid.pack(anchor="w", pady=5)

        # Apply Button
        self.compress_button = tk.Button(self.tab_compress, text="⚡ Apply Compression", font=("Segoe UI", 11, "bold"), 
                                         bg="#2ecc71", fg="white", relief=tk.FLAT, padx=10, pady=8, command=self.run_compression)
        self.compress_button.pack(fill=tk.X, pady=(10, 0))
        
        self.save_button = tk.Button(self.tab_compress, text="💾 Save Result", font=self.font_normal,
                                     bg="#95a5a6", fg="white", relief=tk.FLAT, padx=10, pady=5, command=self.save_image)
        self.save_button.pack(fill=tk.X, pady=5)

        # Statistics & Chart Area
        tk.Label(self.tab_compress, text="Statistics", font=self.font_header, bg=COLOR_SIDEBAR, fg=COLOR_TEXT_WHITE).pack(anchor="w", pady=(15, 5))
        
        self.stats_frame = tk.Frame(self.tab_compress, bg=COLOR_SIDEBAR)
        self.stats_frame.pack(fill=tk.X)
        self.stats_labels = {}
        
        for stat in ["Algorithm", "Time (ms)", "Nodes", "Ratio"]:
            row = tk.Frame(self.stats_frame, bg=COLOR_SIDEBAR)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=stat, font=("Segoe UI", 9), bg=COLOR_SIDEBAR, fg="#bdc3c7", width=12, anchor="w").pack(side=tk.LEFT)
            lbl = tk.Label(row, text="-", font=("Segoe UI", 9, "bold"), bg=COLOR_SIDEBAR, fg="white", anchor="w")
            lbl.pack(side=tk.LEFT)
            self.stats_labels[stat] = lbl

        tk.Label(self.tab_compress, text="File Size (KB)", font=("Segoe UI", 10, "bold"), bg=COLOR_SIDEBAR, fg=COLOR_TEXT_WHITE).pack(anchor="w", pady=(15, 5))
        self.chart_canvas = tk.Canvas(self.tab_compress, bg=COLOR_SIDEBAR, height=60, highlightthickness=0)
        self.chart_canvas.pack(fill=tk.X)


        # --- Main Content (Right) ---
        main_area = tk.Frame(self.root, bg=COLOR_BG)
        main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        self.image_canvas = tk.Canvas(main_area, bg="#bdc3c7", highlightthickness=0)
        self.image_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Events for Cropping
        self.image_canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.image_canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.image_canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

        self.compare_button = tk.Button(main_area, text="👁 Hold to View Original", font=self.font_normal,
                                        bg="#95a5a6", fg="white", relief=tk.FLAT, cursor="hand2")
        self.compare_button.pack(fill=tk.X, pady=(5, 10))
        self.compare_button.bind('<ButtonPress-1>', self.show_original)
        self.compare_button.bind('<ButtonRelease-1>', self.show_compressed)

        # NEW: Progress Bar
        self.progress_bar = ttk.Progressbar(main_area, mode='indeterminate', length=200)
        self.progress_bar.pack(fill=tk.X, pady=(0, 5)) # Reduced padding
        self.progress_bar.pack_forget() # Hide initially

        # NEW: Progress Status Label (Replaces active timer)
        self.progress_status_label = tk.Label(main_area, text="", font=("Segoe UI", 9), bg=COLOR_BG, fg=COLOR_TEXT_DARK)
        self.progress_status_label.pack(fill=tk.X, pady=(0, 5))

        self.status_label = tk.Label(main_area, text="Welcome. Please load an image.", font=("Segoe UI", 9), bg=COLOR_BG, fg=COLOR_TEXT_DARK, anchor="w")
        self.status_label.pack(fill=tk.X, pady=(0, 10))


    def validate_input(self, new_value):
        if not new_value: return True
        try: return 1 <= int(new_value) <= 100
        except ValueError: return False

    def start_progress(self, message="Processing..."):
        """Starts the progress bar visualization."""
        self.progress_bar.pack(fill=tk.X, pady=(0, 5), before=self.progress_status_label)
        self.progress_bar.start(10)
        self.progress_status_label.config(text=message)

    def stop_progress(self):
        """Stops and hides the progress bar visualization."""
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_status_label.config(text="") # Clear status text

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not path: return
        self.original_filepath = path
        self.original_filesize = os.path.getsize(path)
        self.raw_image = Image.open(path).convert("RGB")
        
        # Reset state with new image
        self.base_working_image = self.raw_image.copy() 
        self.current_image = self.base_working_image.copy()
        
        self.status_label.config(text=f"Loaded: {os.path.basename(path)}")
        self.set_controls_state('normal')
        
        # Reset Filters UI
        self.grayscale_var.set(False)
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        
        # Prepare display
        self.display_image_on_canvas(self.current_image)
        
        self.status_label.config(text="Ready.")
        
        # Initial run uses the current image
        self.analyzer = ImageAnalyzer(self.current_image)
        
        # Start thread with progress
        self.start_progress(message="Loading and analyzing image...")
        threading.Thread(target=self.run_compression_thread, daemon=True).start()

    def reset_edits(self):
        """Resets crop and filters to original loaded state."""
        if not self.raw_image: return
        self.base_working_image = self.raw_image.copy()
        self.grayscale_var.set(False)
        self.brightness_var.set(1.0)
        self.contrast_var.set(1.0)
        self.apply_edits() # This will rebuild current_image from base and refresh display
        self.status_label.config(text="Edits Reset.")

    def apply_transform(self, transform_type):
        """Applies rotation or flip to the base working image."""
        if not self.base_working_image: return
        
        self.start_progress(message="Applying transformation...")
        
        def transform_task():
            if transform_type == "rotate_left":
                self.base_working_image = self.base_working_image.transpose(Image.ROTATE_90)
            elif transform_type == "rotate_right":
                self.base_working_image = self.base_working_image.transpose(Image.ROTATE_270)
            elif transform_type == "flip_h":
                self.base_working_image = self.base_working_image.transpose(Image.FLIP_LEFT_RIGHT)
            elif transform_type == "flip_v":
                self.base_working_image = self.base_working_image.transpose(Image.FLIP_TOP_BOTTOM)
                
            # Re-apply filters to the newly transformed base
            # Need to schedule UI update on main thread
            self.root.after(0, self.finish_transform, transform_type)

        threading.Thread(target=transform_task, daemon=True).start()

    def finish_transform(self, transform_type):
        self.apply_edits()  # Rebuild current_image from transformed base and refresh UI.
        self.stop_progress()
        self.status_label.config(text=f"Applied transform: {transform_type}")

    def apply_edits(self):
        """Applies filters (Gray, Bright, Contrast) to the BASE working image (which might be cropped/rotated)."""
        if not self.base_working_image: return
        
        # Start from the cropped base
        img = self.base_working_image.copy()
        
        # 1. Grayscale
        if self.grayscale_var.get():
            img = img.convert("L").convert("RGB")
            
        # 2. Brightness
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(self.brightness_var.get())
        
        # 3. Contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(self.contrast_var.get())
        
        self.current_image = img
        self.display_image_on_canvas(self.current_image)
        
        # Invalidate compressed view since source changed
        self.compressed_image = None 
        self.last_blocks = None

    def run_compression(self):
        if not self.current_image: return
        
        self.status_label.config(text=f"Running compression...")
        self.start_progress(message="Compressing image, please wait...")
        
        # Run heavy lifting in a thread
        threading.Thread(target=self.run_compression_thread, daemon=True).start()

    def run_compression_thread(self):
        # Re-analyze because current_image might have changed due to filters/crop
        # This can be slow for large images, so it's good it's in the thread now
        self.analyzer = ImageAnalyzer(self.current_image)
        
        algo = self.selected_algo.get()
        try: threshold = int(self.slider_var.get())
        except: threshold = 1
        
        start_time = time.time()
        blocks = []
        count = 0

        if algo == "Uniform Grid":
            block_size = int(1 + (threshold / 100) * 63)
            blocks, count = compress_uniform_grid(self.analyzer, self.current_image.size, block_size)
            self.root.after(0, lambda: self.slider_info_label.config(text=f"Block Size: {block_size}px"))
        elif algo == "Quadtree":
            percent = (threshold - 1) / 99.0
            qt_threshold = int(10 + percent * 190)
            blocks, count = compress_quadtree(self.analyzer, self.current_image.size, qt_threshold)
            self.root.after(0, lambda: self.slider_info_label.config(text=f"Variance Threshold: {qt_threshold}"))
        elif algo == "k-d Tree":
            percent = (threshold - 1) / 99.0
            kdt_threshold = int(10 + percent * 190)
            blocks, count = compress_kdtree(self.analyzer, self.current_image.size, kdt_threshold)
            self.root.after(0, lambda: self.slider_info_label.config(text=f"Variance Threshold: {kdt_threshold}"))
        
        end_time = time.time()
        dt = (end_time - start_time) * 1000
        
        # Schedule UI updates on the main thread
        self.root.after(0, lambda: self.finish_compression(algo, dt, count, blocks))

    def finish_compression(self, algo, dt, count, blocks):
        self.stop_progress() # Stops progress bar and status message
        self.last_blocks = blocks 
        self.draw_compressed_image(blocks)
        self.update_statistics(algo, dt, count)
        self.status_label.config(text=f"Finished {algo} in {dt:.2f}ms")

    def refresh_display(self):
        # Only redraws the grid lines if we have last compressed data
        if self.last_blocks and self.current_image:
            self.draw_compressed_image(self.last_blocks)

    def draw_compressed_image(self, blocks):
        if not self.current_image: return
        # Create new image based on current source dimensions
        self.display_image_obj = Image.new("RGB", self.current_image.size)
        draw = ImageDraw.Draw(self.display_image_obj)
        
        # Check grid setting
        outline = "red" if self.show_grid_var.get() else None
        
        for box, color in blocks: 
            draw.rectangle(box, fill=color, outline=outline)
            
        self.display_image_on_canvas(self.display_image_obj, is_compressed=True)

    def display_image_on_canvas(self, img, is_compressed=False):
        cw, ch = self.image_canvas.winfo_width(), self.image_canvas.winfo_height()
        if cw < 10: cw, ch = 800, 600 # Fallback
        
        iw, ih = img.size
        self.img_ratio = min(cw / iw, ch / ih)
        new_size = (int(iw * self.img_ratio), int(ih * self.img_ratio))
        
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        self.image_tk = ImageTk.PhotoImage(resized)
        
        # If displaying compressed result, we need to store the CURRENT edited source as "Original" for comparison
        if is_compressed and self.current_image:
            resized_orig = self.current_image.resize(new_size, Image.Resampling.LANCZOS)
            self.original_tk = ImageTk.PhotoImage(resized_orig)
        elif not is_compressed:
            # If we are just editing/viewing the source, set it as the "original" too
            self.original_tk = self.image_tk

        self.image_canvas.delete("all")
        self.image_canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self.image_tk)

    def show_original(self, event):
        if self.original_tk:
            cw, ch = self.image_canvas.winfo_width(), self.image_canvas.winfo_height()
            self.image_canvas.delete("all")
            self.image_canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self.original_tk)
            self.compare_button.config(text="👁 Viewing Source", bg=COLOR_ACCENT)

    def show_compressed(self, event):
        if self.image_tk:
            cw, ch = self.image_canvas.winfo_width(), self.image_canvas.winfo_height()
            self.image_canvas.delete("all")
            self.image_canvas.create_image(cw // 2, ch // 2, anchor=tk.CENTER, image=self.image_tk)
            self.compare_button.config(text="👁 Hold to View Source", bg="#95a5a6")

    # --- CROP LOGIC ---
    def start_crop_mode(self):
        if not self.current_image: return
        self.is_cropping = True
        self.status_label.config(text="CROP MODE: Drag a box on the image. Click again to cancel.")
        self.image_canvas.config(cursor="crosshair")

    def on_canvas_click(self, event):
        if not self.is_cropping: return
        self.crop_start = (event.x, event.y)
        if self.crop_rect_id: self.image_canvas.delete(self.crop_rect_id)
        self.crop_rect_id = self.image_canvas.create_rectangle(event.x, event.y, event.x, event.y, outline=COLOR_CROP_OUTLINE, width=2)

    def on_canvas_drag(self, event):
        if not self.is_cropping or not self.crop_start: return
        self.image_canvas.coords(self.crop_rect_id, self.crop_start[0], self.crop_start[1], event.x, event.y)

    def on_canvas_release(self, event):
        if not self.is_cropping or not self.crop_start: return
        
        # Calculate coordinates relative to the displayed image center
        cw, ch = self.image_canvas.winfo_width(), self.image_canvas.winfo_height()
        iw, ih = self.current_image.size
        new_w, new_h = int(iw * self.img_ratio), int(ih * self.img_ratio)
        
        # Offset of the image top-left corner on the canvas
        off_x = (cw - new_w) // 2
        off_y = (ch - new_h) // 2
        
        # Mouse coordinates
        x1, y1 = self.crop_start
        x2, y2 = event.x, event.y
        
        # Map canvas coords to image coords
        real_x1 = int((x1 - off_x) / self.img_ratio)
        real_y1 = int((y1 - off_y) / self.img_ratio)
        real_x2 = int((x2 - off_x) / self.img_ratio)
        real_y2 = int((y2 - off_y) / self.img_ratio)
        
        # Normalize (ensure start < end) and Clamp to image bounds
        rx1, rx2 = sorted([real_x1, real_x2])
        ry1, ry2 = sorted([real_y1, real_y2])
        
        rx1 = max(0, rx1); ry1 = max(0, ry1)
        rx2 = min(iw, rx2); ry2 = min(ih, ry2)
        
        if (rx2 - rx1) > 10 and (ry2 - ry1) > 10:
            if messagebox.askyesno("Confirm Crop", "Crop to selected area?"):
                # Crop the BASE working image
                self.base_working_image = self.base_working_image.crop((rx1, ry1, rx2, ry2))
                # Re-apply filters to the new cropped base
                self.apply_edits()
                self.status_label.config(text="Crop applied.")
        
        self.is_cropping = False
        self.image_canvas.config(cursor="")
        self.image_canvas.delete(self.crop_rect_id)

    def update_statistics(self, algo, time_ms, count):
        self.stats_labels["Algorithm"].config(text=algo)
        self.stats_labels["Time (ms)"].config(text=f"{time_ms:.2f}")
        self.stats_labels["Nodes"].config(text=f"{count}")
        
        compressed_size = 0
        if self.display_image_obj:
            buffer = io.BytesIO()
            self.display_image_obj.save(buffer, "PNG")
            compressed_size = buffer.tell()
            if self.original_filesize > 0:
                self.stats_labels["Ratio"].config(text=f"{(compressed_size / self.original_filesize) * 100:.1f}%")

        # Draw Comparison Chart
        self.draw_chart(self.original_filesize, compressed_size)

    def draw_chart(self, orig_size, comp_size):
        self.chart_canvas.delete("all")
        w = self.chart_canvas.winfo_width()
        if w < 10: w = 280
        h = 60
        max_val = max(orig_size, comp_size)
        if max_val == 0: return

        # Reduce scale width (w - 100 instead of w - 60) to prevent text clipping
        max_bar_width = w - 100

        # Bar 1: Original
        len_orig = (orig_size / max_val) * max_bar_width
        self.chart_canvas.create_text(5, 15, text="Orig", fill="white", anchor="w", font=("Segoe UI", 8))
        self.chart_canvas.create_rectangle(40, 5, 40 + len_orig, 25, fill=COLOR_BAR_ORIGINAL, outline="")
        self.chart_canvas.create_text(45 + len_orig, 15, text=f"{orig_size/1024:.0f}K", fill="white", anchor="w", font=("Segoe UI", 8))

        # Bar 2: Compressed
        len_comp = (comp_size / max_val) * max_bar_width
        self.chart_canvas.create_text(5, 45, text="Comp", fill="white", anchor="w", font=("Segoe UI", 8))
        self.chart_canvas.create_rectangle(40, 35, 40 + len_comp, 55, fill=COLOR_BAR_COMPRESSED, outline="")
        self.chart_canvas.create_text(45 + len_comp, 45, text=f"{comp_size/1024:.0f}K", fill="white", anchor="w", font=("Segoe UI", 8))

    def save_image(self):
        if not self.display_image_obj: return
        name, _ = os.path.splitext(os.path.basename(self.original_filepath))
        save_path = filedialog.asksaveasfilename(initialfile=f"{name}_compressed.png", defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            self.display_image_obj.save(save_path)
            self.status_label.config(text=f"Saved to {os.path.basename(save_path)}")
