import tensorflow as tf
import numpy as np

TRAFFIC_LIGHT_CLASS_ID  = 10
CONFIDENCE_CUTOFF       = 0.7
GRAPH_FILE_PATH = "./frozen_detector_graph.pb"


class ObjectDetector(object):
    """
    Represents a general object detector with methods dedicated to traffic light detection.
    Pretrained Tensorflow's Faster RCNN is used as a detector. Network accepts one arbitrary shaped image of uint8
    i.e. (1, None, None, 3, dtype=tf.uint8)
    """
    def __init__(self):
        # graph
        self.detection_graph   = None
        # input image placeholder (1, None, None, 3)
        self.image_tensor      = None
        # result of detection (1, x, 4), where 4 numbers are top, left, bottom, right coordinates of detected image
        # area in the form of [0..1] of image widht/height
        self.detection_boxes   = None
        # result of detection (1, x)
        self.detection_scores  = None
        # detected classes (1, x)
        self.detection_classes = None

        self.detection_graph, \
        self.image_tensor, \
        self.detection_boxes, \
        self.detection_scores, \
        self.detection_classes = self.get_graph_and_tensors(GRAPH_FILE_PATH)

        # create session
        self.session = tf.Session(graph=self.detection_graph)

    def load_graph(self, graph_file_path):
        """
        Loads frozen graph.
        :param graph_file_path: file path to graph
        :return: graph
        """
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name="")
        return graph

    def get_graph_and_tensors(self, graph_file_path):
        """
        Loads graph and retrieves needed tensors.
        :param graph_file_path: graph file path
        :return: (graph, image placeholder, boxes, scores, classes)
        """
        detection_graph = self.load_graph(graph_file_path)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        image_tensor = detection_graph.get_tensor_by_name("image_tensor:0")

        # Each box represents a part of the image where a particular object was detected.
        detection_boxes = detection_graph.get_tensor_by_name("detection_boxes:0")

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        detection_scores = detection_graph.get_tensor_by_name("detection_scores:0")

        # The classification of the object (integer id).
        detection_classes = detection_graph.get_tensor_by_name("detection_classes:0")
        return detection_graph, image_tensor, detection_boxes, detection_scores, detection_classes

    def filter(self, min_score, class_id, boxes, scores, classes):
        """
        Filters detection results by score and class
        :param min_score: keep only results with score higher than this
        :param class_id: keep only results of this class
        :param boxes: to filter
        :param scores: to filter
        :param classes: to filter
        :return: filtered boxes, filtered scores, filtered classes
        """
        score_idx = scores >= min_score
        class_idx = classes == class_id
        indices = score_idx & class_idx

        filtered_boxes = boxes[indices, ...]
        filtered_scores = scores[indices, ...]
        filtered_classes = classes[indices, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def detect_traffic_lights(self, image):
        """
        Detects traffic lights in the given image and returns list of boxes.
        :param image: image to do detection on
        :return: np array (x, 4) of top, left, bottom, right coordinates in pixels
        """
        # do detection
        image_batch = np.expand_dims(image, 0)
        (boxes, scores, classes) = self.session.run([self.detection_boxes,
                                                     self.detection_scores,
                                                     self.detection_classes],
                                                    feed_dict={self.image_tensor: image_batch})

        # remove first redundant dimension
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int)

        # Keep only high confidence traffic light results
        boxes, scores, classes = self.filter(CONFIDENCE_CUTOFF, TRAFFIC_LIGHT_CLASS_ID, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        height, width, _ = image.shape
        box_coords = self.to_image_coords(boxes, height, width).astype(np.int32)

        return box_coords