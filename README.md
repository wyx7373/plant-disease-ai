# plant-disease-ai


## Algorithm
This AI is built with the ImageNet classification trained on thousands of plant photo data using Resnet-18 and PyTorch.

## Setup
Pull the repository to the Jetson Nano home directory. Make sure the jetson-inference and pytorch libraries are installed.

## Train
Make sure the current directory is the repository folder.

Run the train.py file in the repository folder using the following command:

`
python3 train.py--model-dir=models/garbage_classification data/garbage-data-main
`
