# TensorFlow_Flask
Deploy TensorFlow Model using Flask in Python

development environment：

Ubuntu 18.04

Python 3.6.8 

Tensorflow 1.14.0

First of all ,to make sure the TensorFlow Object Detection API works.(https://github.com/tensorflow/models)

let’s start with a tweaked version of the official the Object Detection Demo Jupyter Notebook(models/research/object_detection/object_detection_tutorial.ipynb).  I saved this file as run_original.py(models/research/object_detection/run_original.ipynb).

Go to your favorite  browser and put your URL in.http://0.0.0.0:8080/test

you could upload a picture and detect it.

URL=http://0.0.0.0:8080/

you will see  "hello"

run_web_return_pic.py

you will see a picture in your browser

run_web_return_json.py(models/research/object_detection/run_web_return_json.ipynb)

you will see json string in your browser


Dependencies

pip3 install matplotlib

pip3 install Pillow

pip3 install flask

pip3 install opencv-python

pip3 install requests
