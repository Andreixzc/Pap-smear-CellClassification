import os
import glob
from tkinter import Tk, Label, Button, LEFT, RIGHT, TOP, BOTTOM, Frame
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from hu_moments_extractor import HuMomentsExtractor

class ImageViewer:
    def __init__(self, root, directories):
        self.root = root
        self.root.title("Image Viewer")
        
        self.frame_left = Frame(root)
        self.frame_left.pack(side=LEFT, fill="both", expand=True)
        
        self.frame_right = Frame(root)
        self.frame_right.pack(side=RIGHT, fill="both", expand=True)
        
        self.image_label = Label(self.frame_left)
        self.image_label.pack()
        
        self.prev_button = Button(self.frame_left, text="Previous", command=self.prev_image)
        self.prev_button.pack(side=LEFT)
        
        self.next_button = Button(self.frame_left, text="Next", command=self.next_image)
        self.next_button.pack(side=RIGHT)
        
        self.gray_button = Button(self.frame_left, text="Gray Scale", command=self.show_gray_image)
        self.gray_button.pack(side=TOP)
        
        self.original_button = Button(self.frame_left, text="Original", command=self.show_image)
        self.original_button.pack(side=BOTTOM)
        
        self.zoom_in_button = Button(self.frame_left, text="Zoom In", command=lambda: self.zoom_image(1.2))
        self.zoom_in_button.pack(side=TOP)
        
        self.zoom_out_button = Button(self.frame_left, text="Zoom Out", command=lambda: self.zoom_image(0.8))
        self.zoom_out_button.pack(side=TOP)
        
        self.hu_moments_label = Label(self.frame_right, text="Hu Moments:")
        self.hu_moments_label.pack()
        
        self.directories = directories
        self.image_paths = self.get_image_paths(directories)
        
        if not self.image_paths:
            print("No images found in the provided directories.")
            return
        
        self.current_image_index = 0
        self.current_canvas = None
        self.zoom_scale = 1.0
        self.show_image()
        
    def get_image_paths(self, directories):
        image_paths = []
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        for directory in directories:
            abs_directory = os.path.join(base_dir, directory)
            if not os.path.isdir(abs_directory):
                print(f"Directory does not exist: {abs_directory}")
                continue
            found_images = glob.glob(os.path.join(abs_directory, '*.*'))
            if found_images:
                image_paths.extend(found_images)
            else:
                print(f"No images found in directory: {abs_directory}")
        return sorted(image_paths)

    def show_image(self, index=None):
        if index is not None:
            self.current_image_index = index
        image_path = self.image_paths[self.current_image_index]
        self.image = Image.open(image_path)
        self.display_image = self.image.resize((int(800 * self.zoom_scale), int(600 * self.zoom_scale)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)
        
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo
        
        self.show_histograms()
        self.show_hu_moments(image_path)

    def show_gray_image(self):
        image = self.image.convert("L")
        self.display_image = image.resize((int(800 * self.zoom_scale), int(600 * self.zoom_scale)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display_image)
        
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo
        
        self.show_histograms()
        self.show_hu_moments(self.image_paths[self.current_image_index])

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.zoom_scale = 1.0
            self.show_image(self.current_image_index)
    
    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
            self.zoom_scale = 1.0
            self.show_image(self.current_image_index)
    
    def zoom_image(self, scale_factor):
        self.zoom_scale *= scale_factor
        self.show_image(self.current_image_index)
    
    def show_histograms(self):
        gray_image = self.image.convert("L")
        gray_array = np.array(gray_image)
        
        hsv_image = cv2.cvtColor(np.array(self.image), cv2.COLOR_RGB2HSV)
        h_hist = cv2.calcHist([hsv_image], [0], None, [256], [0, 256])
        s_hist = cv2.calcHist([hsv_image], [1], None, [256], [0, 256])
        v_hist = cv2.calcHist([hsv_image], [2], None, [256], [0, 256])
        
        if self.current_canvas:
            self.current_canvas.get_tk_widget().destroy()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
        
        ax1.hist(gray_array.ravel(), bins=256, range=(0, 256), color='gray')
        ax1.set_title("Gray Histogram")
        ax1.set_xlabel("Pixel Value")
        ax1.set_ylabel("Frequency")
        
        ax2.plot(h_hist, color='r', label='H channel')
        ax2.plot(s_hist, color='g', label='S channel')
        ax2.plot(v_hist, color='b', label='V channel')
        ax2.set_title("HSV Histogram")
        ax2.set_xlabel("Pixel Value")
        ax2.set_ylabel("Frequency")
        ax2.legend()
        
        plt.tight_layout()
        
        self.current_canvas = FigureCanvasTkAgg(fig, master=self.frame_right)
        self.current_canvas.draw()
        self.current_canvas.get_tk_widget().pack(fill="both", expand=True)
        plt.close()

    def show_hu_moments(self, image_path):
        image = cv2.imread(image_path)
        if image is not None:
            extractor = HuMomentsExtractor()
            hu_moments = extractor.extract_hu_moments(image)
            hu_text = "\n".join([f"{i+1}: {moment:.5e}" for i, moment in enumerate(hu_moments)])
            self.hu_moments_label.config(text=f"Hu Moments:\n{hu_text}")
        else:
            self.hu_moments_label.config(text="Hu Moments: Error loading image")

if __name__ == "__main__":
    directories = [
        "28-05-2024/ASC-H", 
        "28-05-2024/ASC-US", 
        "28-05-2024/HSIL", 
        "28-05-2024/LSIL", 
        "28-05-2024/Negative for intraepithelial lesion", 
        "28-05-2024/SCC"
    ]
    
    root = Tk()
    viewer = ImageViewer(root, directories)
    root.mainloop()
