from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import scipy.misc

GRAPH_FILE_PATH = "./frozen_classifier_graph.pb"

RED_CLASS       = 0
YELLOW_CLASS    = 1
GREEN_CLASS     = 2
UNKNOWN_CLASS   = 3

class TLClassifier:
    """
    Traffic light classifier. Classifies given image as being red/yellow/green/uknown.
    The network is a manually trained Bosch classification network.
    It accepts batch of images (None, 64, 64, 3, dtype=np.float32) where each image
    is preprocessed as (image/255.) - 0.5

    """
    def __init__(self):
        # graph
        self.classification_graph = None
        # input image placeholder, (None, 64, 64, 3), should be preprocessed (/255) - 0.5
        self.image_tensor         = None
        # keep prob placeholder, should be 1.
        self.keep_prob_tensor     = None
        # tensor of (None) shape, contains results
        self.predictions_tensor   = None

        self.classification_graph, \
        self.image_tensor, \
        self.keep_prob_tensor, \
        self.predictions_tensor = self.get_graph_and_tensors(GRAPH_FILE_PATH)

        self.session = tf.Session(graph=self.classification_graph)

    def load_graph(self, graph_file_path):
        """
        Loads frozen graph.
        :param graph_file_path: graph file
        :return: graph
        """
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def get_graph_and_tensors(self, graph_file_path):
        """
        Loads graph and retrieves needed tensors.
        :param graph_file_path: graph file path
        :return: graph, image placeholder, keep prob placeholder, predictions tensor
        """
        detection_graph = self.load_graph(graph_file_path)
        # detection_graph = load_graph(RFCN_GRAPH_FILE)
        # detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        image_tensor = detection_graph.get_tensor_by_name('input_image:0')

        # Each box represents a part of the image where a particular object was detected.
        keep_prob_tensor = detection_graph.get_tensor_by_name('keep_prob:0')

        predictions_tensor = detection_graph.get_tensor_by_name("predictions:0")

        return detection_graph, image_tensor, keep_prob_tensor, predictions_tensor

    def get_classification(self, images):
        """
        Does classification on given images. Returns red if red classification is present,
        othewise yellow if yellow is present, otherwise green if green is present, othewise uknonw result.
        :param images: list of images of arbitrary sizes
        :return: TrafficLight.state
        """
        # 1. resize images to (x, 64, 64, 3)
        images_np = np.zeros((len(images), 64, 64, 3), dtype=np.float32)
        for i, image in enumerate(images):
            images_np[i] = ((scipy.misc.imresize(image, (64, 64), interp="bicubic"))/255.) - 0.5

        # 2. do classification
        classifications = self.session.run(self.predictions_tensor, feed_dict={self.image_tensor: images_np, self.keep_prob_tensor: 1.})

        # 3. calculate state
        has_red     = False
        has_yellow  = False
        has_green   = False

        for cl in classifications:
            has_red = has_red | (cl == RED_CLASS)
            has_yellow = has_yellow | (cl == YELLOW_CLASS)
            has_green = has_green | (cl == GREEN_CLASS)

        state = TrafficLight.RED if has_red else \
               (TrafficLight.YELLOW if has_yellow else
               (TrafficLight.GREEN if has_green else
                TrafficLight.UNKNOWN))

        return state, classifications
