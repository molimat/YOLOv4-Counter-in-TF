import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.sort import *
import core.roi as roi
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-416',
                    'path to weights file')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/road.mp4', 'path to input video')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def main(_argv):
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    input_size = FLAGS.size
    video_path = FLAGS.video

    print("Video from: ", video_path )
    vid = cv2.VideoCapture(video_path)

    
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_number = 0
    frame_count = vid.get(7)
    vid.set(1, frame_count/2)
    _, frame_to_roi = vid.read() #frame to be sent to roi, a random.
    frame_to_roi = Image.fromarray(frame_to_roi)
    vid.set(1, 1) # back to begin

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('output' + str(round(time.time()))+ '.avi', fourcc, fps, (width, height))

    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(200, 3),dtype="uint8")

    # initialize our tracker
    tracker = Sort()
    memory = {}


    #ROI
    # line = [(0, int(round(height*0.8))), (width, int(round(height*0.8)))]
    roi_line = roi.get_ROI_line(frame_to_roi)
    line = [roi_line[0], roi_line[1]]
    counter = 0
    print("ROI initialized on %s"%(line))

    if FLAGS.framework == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)
    else:
        saved_model_loaded = tf.saved_model.load(FLAGS.weights, tags=[tag_constants.SERVING])
        infer = saved_model_loaded.signatures['serving_default']

    while True:
        return_value, frame = vid.read()
        if return_value:
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            raise ValueError("No image! Try with another video format")
        frame_size = frame.shape[:2]
        image_data = cv2.resize(frame, (input_size, input_size))
        image_data = image_data / 255.
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        if FLAGS.framework == 'tflite':
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
            if FLAGS.model == 'yolov4' and FLAGS.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25)
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25)
        else:
            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score
        )

        bboxes = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]


        # #bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        # dets = []
        # if len(bboxes) > 0:
        #     # loop over the indexes we are keeping
        #     for i, bbox in enumerate(bboxes):
        #         (xmin, ymin, xmax, ymax, classid) = (bbox[0], bbox[1], bbox[2], bbox[3], classes_ids[i])
        #         dets.append([xmin, ymin, xmax, ymax, classid])
        
        # np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
        # dets = np.asarray(dets) 

        dets = utils.prepare_for_tracking(frame, bboxes)
        tracks = tracker.update(dets)

        tboxes = []
        indexIDs = []
        c = []
        previous = memory.copy()
        memory = {}

        for track in tracks:
            tboxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track[4]))
            memory[indexIDs[-1]] = tboxes[-1]

        if len(tboxes) > 0:
            i = int(0)
            for box in tboxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))

                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
                cv2.rectangle(frame, (x, y), (w, h), color, 2)

                if indexIDs[i] in previous:
                    previous_box = previous[indexIDs[i]]
                    (x2, y2) = (int(previous_box[0]), int(previous_box[1]))
                    (w2, h2) = (int(previous_box[2]), int(previous_box[3]))
                    p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
                    p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
                    cv2.line(frame, p0, p1, color, 3)

                    if utils.intersect(p0, p1, line[0], line[1]):
                        counter += 1

                #text = "{}: {:.4f}".format(LABELS[scores[i]], classes[i])
                text = "{}".format(indexIDs[i])
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                i += 1

        # draw line
        cv2.line(frame, line[0], line[1], (0, 255, 255), 5)

        # draw counter
        cv2.putText(frame, str(counter), (100,200), cv2.FONT_HERSHEY_DUPLEX, 5.0, (0, 255, 255), 10)
        # counter += 1


        # image = utils.draw_bbox(frame, bboxes)
        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        print(info)
        # # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # # cv2.imshow("result", result)
        
        output_movie.write(frame)
        frame_number += 1
        print ("writing frame " + str(frame_number))

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    vid.release()   
    output_movie.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
