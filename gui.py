import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

def imageNet(input_URI, output_URI):
	import jetson.inference, jetson.utils

	test_argv = ['imagenet.py', '--model=models/resnet18.onnx', '--input_blob=input_0', '--output_blob=output_0', '--labels=data/labels.txt', input_URI, 'output.JPG']

	# load the recognition network
	net = jetson.inference.imageNet("resnet18", test_argv)
	
	print("input")
	# create video sources & outputs
	input = jetson.utils.videoSource(input_URI, argv=test_argv)
	print("output")
	output = jetson.utils.videoOutput(output_URI, argv=test_argv + ['--deviceType=d'])
	print("font")
	font = jetson.utils.cudaFont()

	output_dict = {
		"Apple___Apple_scab": "Apple Scab",
		"Apple___Black_rot": "Apple Black Rot",
		"Apple___Cedar_apple_rust": "Cedar Apple Rust",
		"Apple___healthy": "Healthy Apple",
		"Cherry_(including_sour)___healthy": "Healthy Cherry",
		"Cherry_(including_sour)___Powdery_mildew": "Powdery Mildew Cherry",
		"Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Corn Gray Leaf Spot",
		"Corn_(maize)___Common_rust_": "Corn Common Rust",
		"Corn_(maize)___healthy": "Healthy Corn",
		"Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight Corn",
		"Grape___Black_rot": "Grape Black Rot",
		"Grape___Esca_(Black_Measles)": "Grape Black Measles",
		"Grape___healthy": "Healthy Grape",
		"Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Grape Leaf Blight",
		"Peach___Bacterial_spot": "Peach Bacterial Spot",
		"Peach___healthy": "Healthy Peach",
		"Pepper,_bell___Bacterial_spot": "Bacterial Spot Bell Pepper",
		"Pepper,_bell___healthy": "Healthy Bell Pepper",
		"Potato___Early_blight": "Early Blight Potato",
		"Potato___healthy": "Healthy Potato",
		"Potato___Late_blight": "Late Blight Potato",
		"Strawberry___healthy": "Healthy Strawberry",
		"Strawberry___Leaf_scorch": "Leaf Scorch Strawberry",
		"Tomato___Bacterial_spot": "Bacterial Spot Tomato",
		"Tomato___Early_blight": "Early Blight Tomato",
		"Tomato___healthy": "Healthy Tomato",
		"Tomato___Late_blight": "Late Blight Tomato",
		"Tomato___Leaf_Mold": "Tomato Leaf Mold",
		"Tomato___Septoria_leaf_spot": "Septoria Leaf Spot Tomato",
		"Tomato___Spider_mites Two-spotted_spider_mite": "Spider Mites Tomato",
		"Tomato___Target_Spot": "Target Spot Tomato",
		"Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
		"Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Tomato Yellow Leaf Curl Virus",
	}

	# capture the next image
	img = input.Capture()

	# classify the image
	class_id, confidence = net.Classify(img)

	# find the object description
	class_desc = net.GetClassDesc(class_id)

	# allocate the output, with half the size of the input
	imgOutput = jetson.utils.cudaAllocMapped(width=img.width * 2, 
	                                 height=img.height * 2, 
	                                 format=img.format)

	jetson.utils.cudaResize(img, imgOutput)

	# overlay the result on the image	
	font.OverlayText(imgOutput, 0, 0, "{:05.2f}% confident".format(confidence * 100), 5, 5, font.White, font.Gray40)
	font.OverlayText(imgOutput, 0, 0, "{}".format(output_dict[class_desc]), 5, 45, font.White, font.Gray40)
	
	# render the image
	output.Render(imgOutput)

def process_image():
	input_image_path = input_entry.get()
	output_image_path = "output.jpg"  # Path to save the processed image

	# TODO: Implement your image processing logic here
	# Example: Resizing the image
	imageNet(input_image_path, output_image_path)

	# Display the output image
	output_image = ImageTk.PhotoImage(Image.open(output_image_path))
	output_label.configure(image=output_image)
	output_label.image = output_image

def select_image():
    file_path = filedialog.askopenfilename()
#file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.JPG;*.JPEG;*.PNG")])
    input_entry.delete(0, tk.END)
    input_entry.insert(0, file_path)

# Create the Tkinter window
window = tk.Tk()
window.title("Image Processing")
window.geometry("600x400")

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
