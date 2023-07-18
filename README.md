# plant-disease-ai

## The Algorithm
This AI system utilizes PyTorch, a deep learning framework, along with the "jetson-inference" library. It employs a ResNet-18 model trained on a dataset of plant photos to identify signs of sickness. By leveraging the power of deep learning and the pre-trained ResNet-18 architecture, this AI system can analyze plant images, detect symptoms of sickness, and potentially aid in early diagnosis, providing valuable insights to farmers and researchers in the field of agriculture.

## Setup
Pull the repository to the Jetson Nano home directory. Make sure the jetson-inference and pytorch libraries are installed (as well as the proper dependencies).

## Train
Make sure the current directory is the repository folder.

Run the train.py file in the repository folder using the following command:

`
sudo python3 train.py --model-dir=models/ data/
`

Example command specifying epochs and batch size:

`
sudo python3 train.py --model-dir=models -b=20 --gpu=0 --epochs=25 data
`

## Export the model
Make sure model_best.pth.tar is in the models folder. To export the model, run the onnx_export.py file:

`
python3 onnx_export.py --model-dir=models/
`

## Run the model on a test file
After you have the exported onnx model file, you can run it using a test diseased leaf image using this command:

`
imagenet.py --model=models/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/labels.txt data/test/AppleCedarRust1.JPG output.JPG
`
