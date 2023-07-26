from deepsort.detection_helpers import *
from deepsort.tracking_helpers import *
from  bridge_wrapper import *

# detector = Detector(classes = [2] ) #car
# detector.load_model('yolov7x.pt')

tracker = YOLOv7_DeepSORT(reID_model_path="deepsort/deep_sort/model_weights/mars-small128.pb")
tracker.track_video("video.mp4", output="street.avi", show_live = True, skip_frames = 0, count_objects = True, verbose=1)
