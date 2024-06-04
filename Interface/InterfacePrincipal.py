import os
import glob
from tkinter import Tk, Label, Button, LEFT, RIGHT, TOP, BOTTOM, Frame
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
        
        self.directories = directories
        self.image_paths = self.get_image_paths(directories)
        self.current_image_index = 0
        self.current_canvas = None
        self.show_image()
        
    def get_image_paths(self, directories):
        image_paths = []
        for directory in directories:
            image_paths.extend(glob.glob(os.path.join(directory, '*.*')))
        return sorted(image_paths)

    def show_image(self, index=None):
        if index is not None:
            self.current_image_index = index
        image_path = self.image_paths[self.current_image_index]
        self.image = Image.open(image_path)
        image = self.image.resize((800, 600), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo
        
        self.show_histograms()

    def show_gray_image(self):
        image = self.image.convert("L")
        image = image.resize((800, 600), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        
        self.image_label.config(image=self.photo)
        self.image_label.image = self.photo
        
        self.show_histograms()

    def prev_image(self):
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.show_image(self.current_image_index)
    
    def next_image(self):
        if self.current_image_index < len(self.image_paths) - 1:
            self.current_image_index += 1
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
