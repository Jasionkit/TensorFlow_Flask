
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
import datetime
import base64
import io
import re
#--------------------------------------------------------------------------------------------------------------------------------
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError(
        'Please upgrade your TensorFlow installation to v1.12.*.')





# What model to download.
MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'


# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')



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



def load_image_into_numpy_array(image):
    arr = np.array(image)
    print(arr.shape)
    return arr 



config=tf.ConfigProto()
config.gpu_options.allow_growth=True
sess=tf.Session(graph=detection_graph,config=config)
def get_objects(image, threshold=0.5):

        global sess

        # Definite input and output Tensors for detection_graph
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')



        image_np = load_image_into_numpy_array(image) # change
        im_width, im_height = image.size
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.
        #print(datetime.datetime.now().strftime('predictA %H:%M:%S.%f'))
        oldtime=datetime.datetime.now()
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        newtime=datetime.datetime.now()    
        #print(datetime.datetime.now().strftime('predictB %H:%M:%S.%f'))
        print('%s:ms',(newtime-oldtime))

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


                ymin= float(boxes[c][0])
                xmin= float(boxes[c][1])
                ymax= float(boxes[c][2])
                xmax= float(boxes[c][3])

                (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                  ymin * im_height, ymax * im_height)

                item.left=left
                item.right=right
                item.top=top
                item.bottom=bottom

                output.append(item)

        outputJson = json.dumps([ob.__dict__ for ob in output])
        print(outputJson)
        return outputJson

#--------------------------------------------------------------------------------------------------------------------------------
app = Flask(__name__, static_folder="web_static")
# if we save file to disk, we must use the following configuration.
upload_folder = '/media/ubuntu/models/research/object_detection/test_images/'
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp','JPEG'}
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
    print("request_filename:",request_filename)
    pic = request.files[request_filename]
    img_type = pic.filename.split('.')[1]
    filename = get_random_string(30) + "." + img_type

    filepath = os.path.join(
                            app.config['UPLOAD_FOLDER'],
                            filename)
    pic.save(filepath)
    image = Image.open(filepath)  
    return  image


@app.route("/predict", methods=["POST"])
def predict():
    image1=get_feature_from_client('pic1')
    json_str=get_objects(image1)

    return json_str



@app.route("/predict_base64", methods=["POST"])
def predict_base64():
    a=request.form.get('pic1')

    base64_data = re.sub('^data:image/.+;base64,', '', a)
    byte_data = base64.b64decode(base64_data)
    image_data = BytesIO(byte_data)
    img = Image.open(image_data)
    #img.save('1.jpg')

    json_str=get_objects(img)

    return json_str


def image2array(image):
    (w, h) = image.size
    return np.array(image.getdata()).reshape((h, w, 3)).astype(np.uint8)

def array2image(arr):
    return Image.fromarray(np.uint8(arr))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)
