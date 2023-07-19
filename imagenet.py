#!/usr/bin/env python3
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils

import argparse
import sys


# parse the command line
parser = argparse.ArgumentParser(description="Classify a live camera stream using an image recognition DNN.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.imageNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="googlenet", help="pre-trained model to load (see below for options)")
parser.add_argument("--camera", type=str, default="0", help="index of the MIPI CSI camera to use (e.g. CSI camera 0)\nor for VL42 cameras, the /dev/video device to use.\nby default, MIPI CSI camera 0 will be used.")
parser.add_argument("--width", type=int, default=1280, help="desired width of camera stream (default is 1280 pixels)")
parser.add_argument("--height", type=int, default=720, help="desired height of camera stream (default is 720 pixels)")
parser.add_argument('--headless', action='store_true', default=(), help="run without display")

is_headless = ["--headless"] if sys.argv[0].find('console.py') != -1 else [""]

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)


# load the recognition network
net = jetson.inference.imageNet(opt.network, sys.argv)

# create video sources & outputs
input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv+is_headless)
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

# process frames until the user exits
while True:
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

	# update the title bar
	output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))

	# print out performance info
	net.PrintProfilerTimes()

	# exit on input/output EOS
	if not input.IsStreaming() or not output.IsStreaming():
		break