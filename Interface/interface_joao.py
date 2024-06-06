import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.image_path = ""

        # configure window
        self.title("Trabalho PAI")
        self.geometry(f"{1920}x{1080}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_columnconfigure(3, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # create sidebar frame with widgets
        self.sidebar_frame = ctk.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=5, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="CustomTkinter", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Dropdown menu using tkinter.Menu
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Upload Image", command=self.choose_image)




        self.main_button_1 = ctk.CTkButton(master=self, width=50, height=50, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE"), text="Grayscale")
        self.main_button_1.grid(row=2, column=3, padx=(20, 20), pady=(20, 20), sticky="nsew")
        self.main_button_1.configure(width=10, height=2)
        self.main_button_1.grid_remove()  # Hide the button initially

        # Create an area to display the uploaded image
        self.image_label = ctk.CTkLabel(self)
        self.image_label.grid(row=1, column=1, columnspan=3, padx=(10, 10), pady=(10, 10), sticky="nsew")
        self.image_label.configure(text="")

    def choose_image(self):
        # Open file dialog to select an image file
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.gif")])
        if file_path:
            # Open the image file
            img = Image.open(file_path)
            # Resize the image to fit the window
            img = img.resize((500, 500), Image.LANCZOS)
            # Convert the image to a Tkinter-compatible image
            img_tk = ImageTk.PhotoImage(img)
            # Set the image to the label
            self.image_label.configure(image=img_tk)
            self.image_label.image = img_tk  # Keep a reference to avoid garbage collection
            # Make the button visible after an image is uploaded
            self.main_button_1.grid()  # Show the button

    def change_appearance_mode_event(self, new_mode: str):
        ctk.set_appearance_mode(new_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        ctk.set_widget_scaling(new_scaling_float)

if __name__ == "__main__":
    app = App()
    app.mainloop()
