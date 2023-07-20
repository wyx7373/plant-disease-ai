import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def process_image():
    input_image_path = input_entry.get()
    output_image_path = "output.jpg"  # Path to save the processed image
    
    # TODO: Implement your image processing logic here
    # Example: Resizing the image
    image = Image.open(input_image_path)
    image = image.resize((300, 300))
    image.save(output_image_path)
    
    # Display the output image
    output_image = ImageTk.PhotoImage(Image.open(output_image_path))
    output_label.configure(image=output_image)
    output_label.image = output_image

def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    input_entry.delete(0, tk.END)
    input_entry.insert(0, file_path)

# Create the Tkinter window
window = tk.Tk()
window.title("Image Processing")

# Create the input image selection
input_label = tk.Label(window, text="Input Image:")
input_label.pack()
input_entry = tk.Entry(window)
input_entry.pack()
select_button = tk.Button(window, text="Select Image", command=select_image)
select_button.pack()

# Create the image processing button
process_button = tk.Button(window, text="Process Image", command=process_image)
process_button.pack()

# Create the output image display
output_label = tk.Label(window)
output_label.pack()

# Run the Tkinter event loop
window.mainloop()
