# plant-disease-ai

## The Algorithm
This AI system utilizes PyTorch, a deep learning framework, along with the "jetson-inference" library. It employs a ResNet-18 model trained on a dataset of plant photos to identify signs of sickness. By leveraging the power of deep learning and the pre-trained ResNet-18 architecture, this AI system can analyze plant images, detect symptoms of sickness, and potentially aid in early diagnosis, providing valuable insights to farmers and researchers in the field of agriculture.

## Setup
Pull the repository to the Jetson Nano home directory.

`
git clone https://github.com/wyx7373/plant-disease-ai.git
`

Make sure python (version > 3) and pip3 are installed. Run the following code to download all the dependencies.

Windows:

`
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
`

Ubuntu:

```
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl

sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev

pip3 install Cython

pip3 install numpy==1.19.4 torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```

## Train
Make sure the current directory is the repository folder.

Run the train.py file in the repository folder using the following command:

`
python train.py --model-dir=models/ data/
`

Example command specifying epochs and batch size:

`
python train.py --model-dir=models -b=20 --gpu=0 --epochs=25 data
`

## Export the model
Make sure model_best.pth.tar is in the models folder. To export the model, run the onnx_export.py file:

`
python3 onnx_export.py --model-dir=models/
`

## Run the model on a test file
After you have the exported onnx model file, you can run it using a test diseased leaf image using this command:

`
python3 imagenet.py --model=models/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=data/labels.txt data/test/AppleScab1.JPG output.JPG
`

Example output:

![image](https://github.com/wyx7373/plant-disease-ai/assets/139377134/92f5bd48-16bd-4616-8425-5b8b18d7032d)


## Video for setting up the repository and running the model on a test image:

[![video](https://img.youtube.com/vi/BUD57y0pXgI/0.jpg)](https://www.youtube.com/watch?v=BUD57y0pXgI)

## Video for using the GUI for testing the model:

[![video](https://img.youtube.com/vi/-Wmj4ygaOMY/0.jpg)](https://youtu.be/-Wmj4ygaOMY)
