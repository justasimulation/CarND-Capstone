import rospy
import threading
import time
import numpy as np

from std_msgs.msg import Bool

from PIL import Image, ImageDraw

from light_classification.object_detector import ObjectDetector
from light_classification.tl_classifier import TLClassifier

# traffic light classification result should be constant for at least this number of classifications
# this helps to reduce noise
STATE_COUNT_THRESHOLD = 3

# this number of images will be classified before /traffic_classifier_ready message will be published
# this is needed to warm up the graphic card
WARMUP_NUM = 10


class AsyncPipeline:
    """
    Incapsulates traffic light detector and classifier. Performs detection/classification in a dedicated thread.
    After thread is started some iterations of classification is spent on graphic card warm up, after which a
    message is published to /traffic_classifier_ready topic.
    Methods and variables that can be accessed from different threads have "sh" prefix (shared)
    """
    def __init__(self):
        rospy.loginfo("Initializing traffic light classifier...")
        self.object_detector = ObjectDetector()
        self.classifier = TLClassifier()
        rospy.loginfo("Traffic light classifier is initialized.")

        # this top indicates whether we are ready for detection/classification.
        self.traffic_classifier_ready_pub = rospy.Publisher("/traffic_classifier_ready", Bool, queue_size=1, latch=True)

        # locks for getting/setting image and state values
        self.image_lock = threading.Lock()
        self.state_lock = threading.Lock()


        # fields that are shared between threads
        # should be accessed only via getters/setters which are using locks

        # last camera image
        self.sh_image = None
        # last acknowledged traffic light state (classified enough times in a row, to make sure it is not noise)
        self.sh_acknowledged_state = None

        # result of last classification (may differ from acknowledged state)
        self.sh_last_classified_state = None
        # time of last classification
        self.sh_last_classification_time = 0

    def start_thread(self):
        """
        Starts main classification thread.
        :return:
        """
        thread = threading.Thread(target=self.classification_thread_fn)
        thread.start()

    def sh_set_image(self, image):
        """
        Sets camera image. Thread safe.
        :param image: camera image message
        """
        self.image_lock.acquire()
        self.sh_image = image
        self.image_lock.release()

    def sh_get_image(self):
        """
        Gets camera image thread safe.
        :return: camera image.
        """
        self.image_lock.acquire()
        image = self.sh_image
        self.image_lock.release()
        return image

    def sh_get_state_info(self):
        """
        Returns results of classification. Thread safe.
        :return: acknowledged classification result, last classification result, last classification time
        """
        self.state_lock.acquire()
        acknowledged_state = self.sh_acknowledged_state
        last_classified_state = self.sh_last_classified_state
        last_classification_time = self.sh_last_classification_time
        self.state_lock.release()

        return acknowledged_state, last_classified_state, last_classification_time

    def sh_set_classification_info(self, last_classified_state, last_classification_time):
        """
        Sets last classification information. Thread safe.
        :param last_classified_state: classification result
        :param last_classification_time: classification time
        """
        self.state_lock.acquire()
        self.sh_last_classified_state = last_classified_state
        self.sh_last_classification_time = last_classification_time
        self.state_lock.release()

    def sh_set_state_info(self, acknowledged_state, last_classification_time):
        """
        Sets state information
        :param acknowledged_state: acknowledged classification state
        :param last_classification_time: last classification time
        """
        self.state_lock.acquire()
        self.sh_acknowledged_state = acknowledged_state
        self.sh_last_classified_state = acknowledged_state
        self.sh_last_classification_time = last_classification_time
        self.state_lock.release()

    def classification_thread_fn(self):
        """
        Performs classification in a dedicated thread.
        """
        # last image that classification was done on
        last_image = None
        # last classification result
        last_state = None
        # number of times last classification has the same result in a row
        last_state_count = 0
        # overall count of classifications
        overall_count = 0

        while not rospy.is_shutdown():
            # 1. Get camera image
            image = self.sh_get_image()

            # 2. In case image was changed
            if last_image != image:

                # 3. In case warming up is done send message that the classifier is ready
                if overall_count == WARMUP_NUM:
                    self.traffic_classifier_ready_pub.publish(Bool(True))
                    rospy.loginfo("Warming up is finished")
                elif overall_count < WARMUP_NUM:
                    rospy.loginfo("Warming up")

                # 4. Do classification
                start_time = time.time()
                state = self.get_light_state(image, overall_count)
                classification_time = time.time() - start_time

                if last_state != state:  # in case state is changed, remember it and reset the counter
                    last_state_count = 0
                    last_state = state
                    self.sh_set_classification_info(state, classification_time)
                elif last_state_count >= STATE_COUNT_THRESHOLD:  # in case state persists long enough change it
                    self.sh_set_state_info(last_state, classification_time)
                else: # otherwise just save last classificaton info for debugging
                    self.sh_set_classification_info(state, classification_time)

                last_state_count += 1  # increment how many times in a row we got the same state
                overall_count += 1     # increment how many times we done classification
                last_image = image     # memorize last image

    def get_light_state(self, image, counter):
        """
        Performs detection and then classification.
        :param image: ROS image
        :return: TrafficLight.state
        """
        # 1. do detection
        image_np = np.fromstring(image.data, dtype=np.uint8).reshape((image.height, image.width, 3))
        # site images are in bgr, convert if needed
        if image.encoding == "bgr8":
            image_np = np.flip(image_np, 2)

        boxes = self.object_detector.detect_traffic_lights(image_np)

        # 2. create list of detected subimages
        patches = []
        for top, left, bottom, right in boxes:
            patches.append(image_np[top:bottom, left:right])

        # 3. do classification
        predicted_state, classifications = self.classifier.get_classification(patches)

        #self.draw_boxes(image_np, boxes, classifications, counter)

        return predicted_state

    def draw_boxes(self, image, boxes, classifications, counter, thickness=4):
        """Draw bounding boxes on the image"""
        colors = {0: (255, 0, 0), 1: (255, 255, 0), 2: (0, 255, 0), 3: (255, 255, 255) }

        image = Image.fromarray(image)
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classifications[i])
            color = colors[class_id]
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

        image.save("./tmp/image_{:04d}.png".format(counter))

if __name__ == "__main__":
    cl = AsyncPipeline()

    import glob
    import scipy.misc

    file_paths = glob.glob("/media/heaven6/Just/datasets/udacity_site/just_traffic_light/*.jpg")

    for file_path in file_paths:
        image = scipy.misc.imread(file_path)
        cl.get_light_state(image)






