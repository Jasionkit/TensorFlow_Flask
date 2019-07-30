
from utils import visualization_utils as vis_util
from utils import label_map_util
from object_detection.utils import ops as utils_ops
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


#--------------------------------------------------------------------------------------------------------------------------------
from flask import Flask
from flask import request
from flask import send_file
import cv2
from io import BytesIO
import requests
import json
#--------------------------------------------------------------------------------------------------------------------------------
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError(
        'Please upgrade your TensorFlow installation to v1.12.*.')


# ## Object detection imports
# Here are the imports from the object detection module.


# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.
#
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.


# What model to download.
#MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
MODEL_NAME = 'ssd_mobilenet_v2_oid_v4_2018_12_12'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
#PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('data', 'oid_v4_label_map.pbtxt')

opener = urllib.request.URLopener()
opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
    file_name = os.path.basename(file.name)
    if 'frozen_inference_graph.pb' in file_name:
        tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine


category_index = label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS, use_display_name=True)


# ## Helper code


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(
    PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)



# added to put object in JSON
class Object(object):
    def __init__(self):
        self.name=" TensorFlow Object Detection  API"

    def toJSON(self):
        return json.dumps(self.__dict__)

def get_objects(image, threshold=0.5):
   # with graph.as_default():
   #   with tf.Session() as sess:
    with detection_graph.as_default():
     with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')



        image_np = load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        classes = np.squeeze(classes).astype(np.int32)
        scores = np.squeeze(scores)
        boxes = np.squeeze(boxes)

        obj_above_thresh = sum(n > threshold for n in scores)
        print("detected %s objects in image above a %s score" % (obj_above_thresh, threshold))

        output = []

        # Add some metadata to the output
        item = Object()
        item.version = "0.0.1"
        item.numObjects = int(obj_above_thresh)
        item.threshold = threshold
        output.append(item)

        for c in range(0, len(classes)):
            class_name = category_index[classes[c]]['name']
            if scores[c] >= threshold:      # only return confidences equal or greater than the threshold
                print(" object %s - score: %s, coordinates: %s" % (class_name, scores[c], boxes[c]))

                item = Object()
                item.name = 'Object'
                item.class_name = class_name
                item.score = float(scores[c])
                item.y = float(boxes[c][0])
                item.x = float(boxes[c][1])
                item.height = float(boxes[c][2])
                item.width = float(boxes[c][3])

                output.append(item)

        outputJson = json.dumps([ob.__dict__ for ob in output])
        print(outputJson)
        return outputJson

#--------------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__, static_folder="web_static")
# if we save file to disk, we must use the following configuration.
upload_folder = '~/tensorflow/models/research/object_detection/test_images/'
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
@app.route("/")
def hello():
    return "Hello!"


@app.route("/test")
def test():
    html = '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>object detection</title>
        </head>
        <body>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="pic1" value="Pic1" /><br>
                <input type="submit" value="detect">
            </form>
        </body>
        </html>
    '''
    return html

def get_feature_from_client(request_filename):

    import random
    def get_random_string(length):
        string = ""
        for i in range(0, length):
            code = random.randint(97, 122)
            string += chr(code)
        return string

    pic = request.files[request_filename]
    img_type = pic.filename.split('.')[1]
    filename = get_random_string(30) + "." + img_type
    filepath = os.path.join(app.root_path,
                            app.config['UPLOAD_FOLDER'],
                            filename)
    pic.save(filepath)
    image = Image.open(filepath)  
    return  image


@app.route("/predict", methods=["POST"])
def predict():
    image1= get_feature_from_client('pic1')
    json_str=get_objects(image1)

    return json_str

def image2array(image):
    (w, h) = image.size
    return np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8)

def array2image(arr):
    return Image.fromarray(np.uint8(arr))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=True)
