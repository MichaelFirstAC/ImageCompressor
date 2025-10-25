# COMP6049001 - Algorithm Design and Analysis
# Final Project: A Comparative Analysis of Image Compression Algorithms
# Group Members:
# - Harris Ekaputra Suryadi (2802400502)
# - Michael Arianno Chandrarieta (2802499711)
# - Muhammad Ryan Ismail Putra (2802522733)

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw
import time
import io
import numpy as np
import threading
import os

# --- Performance Optimization: Summed-Area Table (Integral Image) ---
class ImageAnalyzer:
    """
    A helper class to pre-compute summed-area tables for fast region analysis.
    This avoids repeated, slow calculations during the recursive compression process.
    """
    def __init__(self, image):
        # 1. Get RGB data for display color averaging
        self.np_rgb = np.array(image, dtype=np.float32)

        # 2. Get YCbCr data for perceptual variance calculation
        ycbcr_image = image.convert('YCbCr')
        self.np_ycbcr = np.array(ycbcr_image, dtype=np.float32)

        # 3. Pre-compute integral images for both sum and squared sum of YCbCr values
        # This allows for O(1) calculation of mean and variance for any region.
        self.sum_ycbcr = np.pad(self.np_ycbcr.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)), 'constant')
        self.sq_sum_ycbcr = np.pad((self.np_ycbcr ** 2).cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)), 'constant')
        
        # 4. Pre-compute integral image for RGB sum (for fast color averaging)
        self.sum_rgb = np.pad(self.np_rgb.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0), (0, 0)), 'constant')

    def get_region_stats(self, box):
        """
        Calculates avg RGB color and perceptual variance for a box in O(1) time.
        """
        x1, y1, x2, y2 = box
        if x1 >= x2 or y1 >= y2:
            return (0, 0, 0), 0
            
        num_pixels = (x2 - x1) * (y2 - y1)

        # Fast RGB average calculation using the integral image
        rgb_sum = self.sum_rgb[y2, x2] - self.sum_rgb[y1, x2] - self.sum_rgb[y2, x1] + self.sum_rgb[y1, x1]
        avg_color_rgb = tuple((rgb_sum / num_pixels).astype(int))

        # Fast YCbCr sum and squared sum calculation
        ycbcr_sum = self.sum_ycbcr[y2, x2] - self.sum_ycbcr[y1, x2] - self.sum_ycbcr[y2, x1] + self.sum_ycbcr[y1, x1]
        ycbcr_sq_sum = self.sq_sum_ycbcr[y2, x2] - self.sq_sum_ycbcr[y1, x2] - self.sq_sum_ycbcr[y2, x1] + self.sq_sum_ycbcr[y1, x1]

        # Calculate variance from the sums: Var(X) = E[X^2] - (E[X])^2
        mean = ycbcr_sum / num_pixels
        mean_sq = ycbcr_sq_sum / num_pixels
        stds = np.sqrt(mean_sq - mean**2)
        
        # Weighted perceptual variance
        variance = (stds[0] * 2) + stds[1] + stds[2]
        
        return avg_color_rgb, variance

# --- Core Algorithm Implementations (Now using ImageAnalyzer) ---

def compress_uniform_grid(analyzer, image_size, block_size):
    if block_size == 0: return [], 0
    width, height = image_size
    output_blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            box = (x, y, min(x + block_size, width), min(y + block_size, height))
            avg_color, _ = analyzer.get_region_stats(box)
            output_blocks.append((box, avg_color))
    return output_blocks, len(output_blocks)

class QuadtreeNode:
    def __init__(self, box):
        self.box, self.children, self.is_leaf, self.color = box, None, False, None

def build_quadtree(analyzer, box, threshold, depth, max_depth):
    node = QuadtreeNode(box)
    avg_color, variance = analyzer.get_region_stats(box)
    node.color = avg_color
    # Added a minimum block size check (e.g., 4 pixels wide) to prevent excessive recursion.
    if variance < threshold or (box[2] - box[0]) <= 4 or depth >= max_depth:
        node.is_leaf = True
        return node
    x1, y1, x2, y2 = box
    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
    boxes = [(x1, y1, mid_x, mid_y), (mid_x, y1, x2, mid_y), (x1, mid_y, mid_x, y2), (mid_x, mid_y, x2, y2)]
    node.children = [build_quadtree(analyzer, b, threshold, depth + 1, max_depth) for b in boxes if b[0] < b[2] and b[1] < b[3]]
    return node

def get_quadtree_leaves(node):
    if node is None: return []
    if node.is_leaf: return [node]
    leaves = []
    if node.children:
        for child in node.children: leaves.extend(get_quadtree_leaves(child))
    return leaves

def compress_quadtree(analyzer, image_size, threshold):
    max_depth = min(12, int(np.log2(min(image_size))))
    root = build_quadtree(analyzer, (0, 0, image_size[0], image_size[1]), threshold, 0, max_depth)
    leaves = get_quadtree_leaves(root)
    return [(leaf.box, leaf.color) for leaf in leaves], len(leaves)

class KDTreeNode:
    def __init__(self, box):
        self.box, self.left, self.right, self.is_leaf, self.color = box, None, None, False, None

def build_kdtree(analyzer, box, threshold, depth, max_depth):
    node = KDTreeNode(box)
    avg_color, variance = analyzer.get_region_stats(box)
    node.color = avg_color
    
    if variance < threshold or (box[2] - box[0]) <= 4 or (box[3] - box[1]) <= 4 or depth >= max_depth:
        node.is_leaf = True
        return node
        
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    # Reverted to the fast spatial midpoint split
    if (width > height):
        split = (x1 + x2) // 2
        box1, box2 = (x1, y1, split, y2), (split, y1, x2, y2)
    else:
        split = (y1 + y2) // 2
        box1, box2 = (x1, y1, x2, split), (x1, split, x2, y2)

    if box1[0] < box1[2] and box1[1] < box1[3]:
        node.left = build_kdtree(analyzer, box1, threshold, depth + 1, max_depth)
    if box2[0] < box2[2] and box2[1] < box2[3]:
        node.right = build_kdtree(analyzer, box2, threshold, depth + 1, max_depth)
    return node

def get_kdtree_leaves(node):
    if node is None: return []
    if node.is_leaf: return [node]
    leaves = []
    leaves.extend(get_kdtree_leaves(node.left))
    leaves.extend(get_kdtree_leaves(node.right))
    return leaves
    
def compress_kdtree(analyzer, image_size, threshold):
    max_depth = min(14, int(np.log2(min(image_size))) + 4)
    # The 'image' parameter is no longer needed
    root = build_kdtree(analyzer, (0, 0, image_size[0], image_size[1]), threshold, 0, max_depth)
    leaves = get_kdtree_leaves(root)
    return [(leaf.box, leaf.color) for leaf in leaves], len(leaves)

# --- Main Application Class ---
class ImageCompressorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Compression Analyzer")
        self.root.geometry("1400x900")
        self.original_image, self.original_filepath, self.original_filesize = None, None, 0
        self.display_image, self.image_tk, self.jpeg_benchmarks = None, None, {}
        self.slider_var = tk.StringVar(value="25")
        # The debounce timer is no longer needed
        # self._after_id = None
        self.setup_ui()
        self.set_controls_state('disabled')

    def set_controls_state(self, state):
        for widget in self.algo_frame.winfo_children(): widget.config(state=state)
        self.slider.config(state=state)
        self.save_button.config(state=state)
        self.slider_entry.config(state=state)
        # Add the new button to the state control
        self.compress_button.config(state=state)

    def setup_ui(self):
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        top_bar = tk.Frame(main_frame, bg="#d0d0d0")
        top_bar.pack(fill=tk.X, pady=(0, 10))
        self.load_button = tk.Button(top_bar, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT, padx=10, pady=5)
        self.save_button = tk.Button(top_bar, text="Save Compressed Image", command=self.save_image)
        self.save_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.status_label = tk.Label(top_bar, text="Please load an image to begin.", bg="#d0d0d0", fg="black")
        self.status_label.pack(side=tk.LEFT, padx=10)
        content_frame = tk.Frame(main_frame, bg="#f0f0f0")
        content_frame.pack(fill=tk.BOTH, expand=True)
        left_panel = tk.Frame(content_frame, width=350, bg="#e0e0e0")
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        right_panel = tk.Frame(content_frame, bg="white")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.algo_frame = tk.LabelFrame(left_panel, text="Algorithm", padx=10, pady=10, bg="#e0e0e0")
        self.algo_frame.pack(fill=tk.X, padx=10, pady=10)
        self.selected_algo = tk.StringVar(value="Quadtree")
        for text, value in [("Uniform Grid", "Uniform Grid"), ("Quadtree", "Quadtree"), ("k-d Tree", "k-d Tree")]:
            ttk.Radiobutton(self.algo_frame, text=text, variable=self.selected_algo, value=value, command=self.run_compression).pack(anchor=tk.W)
        self.slider_label = tk.Label(left_panel, text="Detail Threshold / Block Size:", bg="#e0e0e0")
        self.slider_label.pack(pady=(10, 0))
        slider_frame = tk.Frame(left_panel, bg="#e0e0e0")
        slider_frame.pack(padx=10)
        self.slider = tk.Scale(slider_frame, from_=1, to=100, orient=tk.HORIZONTAL, length=240, variable=self.slider_var, showvalue=0, bg="#e0e0e0")
        self.slider.pack(side=tk.LEFT)
        validate_cmd = (self.root.register(self.validate_input), '%P')
        self.slider_entry = tk.Entry(slider_frame, textvariable=self.slider_var, width=5, validate='key', validatecommand=validate_cmd)
        self.slider_entry.pack(side=tk.LEFT, padx=(5, 0))
        
        # The trace is removed, so typing/sliding no longer auto-compresses
        # self.slider_var.trace_add("write", self.debounce_run_compression)

        # Add the new "Apply" button
        self.compress_button = tk.Button(left_panel, text="Apply Compression", command=self.run_compression)
        self.compress_button.pack(pady=10)

        stats_frame = tk.LabelFrame(left_panel, text="Statistics", padx=10, pady=10, bg="#e0e0e0")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)
        self.stats_labels = {}
        for stat in ["Algorithm", "Time (ms)", "Blocks/Nodes", "Original Size (KB)", "Compressed Size (KB)", "Compression Ratio (%)"]:
            row = tk.Frame(stats_frame, bg="#e0e0e0")
            row.pack(fill=tk.X)
            tk.Label(row, text=f"{stat}:", anchor="w", width=20, bg="#e0e0e0").pack(side=tk.LEFT)
            self.stats_labels[stat] = tk.Label(row, text="-", anchor="w", bg="#e0e0e0")
            # --- THIS LINE WAS INDENTED INCORRECTLY ---
            # It should be INSIDE the loop.
            self.stats_labels[stat].pack(side=tk.LEFT)
        
        # --- Image Display in Right Panel ---
        # The right panel will now be split top and bottom
        self.image_canvas = tk.Canvas(right_panel, bg="gray", highlightthickness=0)
        self.image_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # --- JPEG Benchmark panel (now horizontal at the bottom) ---
        benchmark_container = tk.LabelFrame(right_panel, text="JPEG Benchmarks", font=("Helvetica", 12), bg="#e0e0e0", padx=5, pady=5)
        benchmark_container.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0))
        
        self.jpeg_canvases = {}
        for quality in [75, 50, 25]:
            # Create a frame for each benchmark so they can be arranged side-by-side
            jpeg_frame = tk.Frame(benchmark_container, bg="#e0e0e0")
            # Use fill and expand to make them share the space equally
            jpeg_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

            tk.Label(jpeg_frame, text=f"Quality: {quality}%", bg="#e0e0e0").pack()
            canvas = tk.Canvas(jpeg_frame, bg="gray", height=200, highlightthickness=0)
            canvas.pack(fill=tk.BOTH, expand=True)
            self.jpeg_canvases[quality] = canvas

    def validate_input(self, new_value):
        if not new_value: return True
        try:
            return 1 <= int(new_value) <= 100
        except ValueError:
            return False

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if not filepath: return
        self.original_filepath = filepath
        self.original_image = Image.open(filepath).convert("RGB")
        self.original_filesize = os.path.getsize(filepath)
        self.stats_labels["Original Size (KB)"].config(text=f"{self.original_filesize / 1024:.2f}")
        self.status_label.config(text=f"Loaded: {os.path.basename(filepath)}")
        self.set_controls_state('normal')
        self.status_label.config(text="Analyzing and processing benchmarks...")
        threading.Thread(target=self.process_benchmarks_and_run, daemon=True).start()
    
    def save_image(self):
        if not self.display_image: return
        name, _ = os.path.splitext(os.path.basename(self.original_filepath))
        save_path = filedialog.asksaveasfilename(initialfile=f"{name}_compressed.png", defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if save_path:
            self.display_image.save(save_path)
            self.status_label.config(text=f"Image saved to {os.path.basename(save_path)}")

    def process_benchmarks_and_run(self):
        self.analyzer = ImageAnalyzer(self.original_image) # Pre-compute tables
        self.generate_jpeg_benchmarks()
        self.root.after(0, self.display_jpeg_benchmarks)
        self.root.after(0, self.run_compression)

    # The debounce function is no longer needed
    # def debounce_run_compression(self, *args):
    #     if self._after_id: self.root.after_cancel(self._after_id)
    #     self._after_id = self.root.after(250, self.run_compression)
        
    def run_compression(self):
        if not self.original_image: return
        algo = self.selected_algo.get()
        try:
            threshold = int(self.slider_var.get())
        except (ValueError, tk.TclError):
            threshold = 1
        self.status_label.config(text=f"Compressing with {algo}...")
        self.root.update_idletasks()
        start_time = time.time()
        
        if algo == "Uniform Grid":
            block_size = int(1 + (threshold / 100) * 63)
            blocks, count = compress_uniform_grid(self.analyzer, self.original_image.size, block_size)
            self.slider_label.config(text=f"Block Size: {block_size}px")
        elif algo == "Quadtree":
            qt_threshold = int(1 + (threshold / 100) * 149)
            blocks, count = compress_quadtree(self.analyzer, self.original_image.size, qt_threshold)
            self.slider_label.config(text=f"Detail Threshold: {qt_threshold}")
        elif algo == "k-d Tree":
            kdt_threshold = int(1 + (threshold / 100) * 149)
            # Reverted to pass image_size instead of the full image object
            blocks, count = compress_kdtree(self.analyzer, self.original_image.size, kdt_threshold)
            self.slider_label.config(text=f"Detail Threshold (k-d): {kdt_threshold}")
        else: return

        end_time = time.time()
        self.draw_compressed_image(blocks)
        self.update_statistics(algo, (end_time - start_time) * 1000, count)
        self.status_label.config(text="Compression complete.")

    def draw_compressed_image(self, blocks):
        if not self.original_image: return
        self.display_image = Image.new("RGB", self.original_image.size)
        draw = ImageDraw.Draw(self.display_image)
        for box, color in blocks: draw.rectangle(box, fill=color)
        canvas_w, canvas_h = self.image_canvas.winfo_width(), self.image_canvas.winfo_height()
        img_w, img_h = self.display_image.size
        ratio = min(canvas_w / img_w, canvas_h / img_h)
        resized_img = self.display_image.resize((int(img_w * ratio), int(img_h * ratio)), Image.Resampling.LANCZOS)
        self.image_tk = ImageTk.PhotoImage(resized_img)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(canvas_w / 2, canvas_h / 2, anchor=tk.CENTER, image=self.image_tk)

    def update_statistics(self, algo, time_ms, count):
        self.stats_labels["Algorithm"].config(text=algo)
        self.stats_labels["Time (ms)"].config(text=f"{time_ms:.2f}")
        self.stats_labels["Blocks/Nodes"].config(text=f"{count}")
        if self.display_image:
            buffer = io.BytesIO()
            self.display_image.save(buffer, "PNG")
            size = buffer.tell()
            self.stats_labels["Compressed Size (KB)"].config(text=f"{size / 1024:.2f}")
            if self.original_filesize > 0:
                self.stats_labels["Compression Ratio (%)"].config(text=f"{(size / self.original_filesize) * 100:.2f}")

    def generate_jpeg_benchmarks(self):
        if not self.original_image: return
        for quality in [75, 50, 25]:
            buffer = io.BytesIO()
            self.original_image.save(buffer, "JPEG", quality=quality)
            buffer.seek(0)
            self.jpeg_benchmarks[quality] = Image.open(buffer)

    def display_jpeg_benchmarks(self):
        for quality, canvas in self.jpeg_canvases.items():
            if quality in self.jpeg_benchmarks:
                img = self.jpeg_benchmarks[quality]
                canvas_w, canvas_h = canvas.winfo_width(), canvas.winfo_height()
                if canvas_w <= 1: continue
                ratio = min(canvas_w / img.width, canvas_h / img.height)
                resized = img.resize((int(img.width * ratio), int(img.height * ratio)), Image.Resampling.LANCZOS)
                img_tk = ImageTk.PhotoImage(resized)
                setattr(self, f"jpeg_tk_{quality}", img_tk)
                canvas.delete("all")
                canvas.create_image(canvas_w / 2, canvas_h / 2, anchor=tk.CENTER, image=img_tk)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageCompressorApp(root)
    root.mainloop()








