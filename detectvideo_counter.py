import time
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import YOLOv4, YOLOv3, YOLOv3_tiny, decode
from PIL import Image
from core.config import cfg
from core import visualization_utils as vis_util
import cv2
import numpy as np
import tensorflow as tf

flags.DEFINE_string('framework', 'tf', '(tf, tflite')
flags.DEFINE_string('weights', './data/yolov4.weights',
                    'path to weights file')
flags.DEFINE_integer('size', 608, 'resize images to')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('video', './data/road.avi', 'path to input video')

def main(_argv):
    if FLAGS.tiny:
        STRIDES = np.array(cfg.YOLO.STRIDES_TINY)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_TINY, FLAGS.tiny)
    else:
        STRIDES = np.array(cfg.YOLO.STRIDES)
        if FLAGS.model == 'yolov4':
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, FLAGS.tiny)
        else:
            ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS_V3, FLAGS.tiny)
    NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
    XYSCALE = cfg.YOLO.XYSCALE
    input_size = FLAGS.size
    video_path = FLAGS.video

    print("Video from: ", video_path )
    vid = cv2.VideoCapture(video_path)
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(vid.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_movie = cv2.VideoWriter('output' + str(round(time.time()))+ '.avi', fourcc, fps, (width, height))
    if FLAGS.framework == 'tf':
        input_layer = tf.keras.layers.Input([input_size, input_size, 3])
        if FLAGS.tiny:
            feature_maps = YOLOv3_tiny(input_layer, NUM_CLASS)
            bbox_tensors = []
            for i, fm in enumerate(feature_maps):
                bbox_tensor = decode(fm, NUM_CLASS, i)
                bbox_tensors.append(bbox_tensor)
            model = tf.keras.Model(input_layer, bbox_tensors)
            utils.load_weights_tiny(model, FLAGS.weights)
        else:
            if FLAGS.model == 'yolov3':
                feature_maps = YOLOv3(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                utils.load_weights_v3(model, FLAGS.weights)
            elif FLAGS.model == 'yolov4':
                feature_maps = YOLOv4(input_layer, NUM_CLASS)
                bbox_tensors = []
                for i, fm in enumerate(feature_maps):
                    bbox_tensor = decode(fm, NUM_CLASS, i)
                    bbox_tensors.append(bbox_tensor)
                model = tf.keras.Model(input_layer, bbox_tensors)
                
                if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
                    utils.load_weights(model, FLAGS.weights)
                else:
                    model.load_weights(FLAGS.weights).expect_partial()

        model.summary()
    else:
        # Load TFLite model and allocate tensors.
        interpreter = tf.lite.Interpreter(model_path=FLAGS.weights)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(input_details)
        print(output_details)

    total_passed_vehicle = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
        else:
            raise ValueError("No image! Try with another video format")
        frame_size = frame.shape[:2]
        image_data = utils.image_preprocess(np.copy(frame), [input_size, input_size])
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        prev_time = time.time()

        if FLAGS.framework == 'tf':
            pred_bbox = model.predict(image_data)
        else:
            interpreter.set_tensor(input_details[0]['index'], image_data)
            interpreter.invoke()
            pred_bbox = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

        if FLAGS.model == 'yolov4':
            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES, XYSCALE)
        else:
            pred_bbox = utils.postprocess_bbbox(pred_bbox, ANCHORS, STRIDES)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.25)
        boxes = bboxes[:, 0:4]
        scores = bboxes[:, 4]
        classes = bboxes[:, 5]
        #bboxes = utils.nms(bboxes, 0.213, method='nms')
        roi = 450
        category_index = utils.read_class_names(cfg.YOLO.CLASSES)
        counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_y_axis(vid.get(1),
                                                                                                            frame,
                                                                                                            1,
                                                                                                            False,
                                                                                                            np.squeeze(boxes),
                                                                                                            np.squeeze(classes).astype(np.int32),
                                                                                                            np.squeeze(scores),
                                                                                                            category_index,
                                                                                                            x_reference = roi,
                                                                                                            use_normalized_coordinates=True,
                                                                                                            line_thickness=4)


        if counter == 1:
            cv2.line(frame, (roi, 0), (roi, height), (0, 0xFF, 0), 5)
        else:
            cv2.line(frame, (roi, 0), (roi, height), (0, 0, 0xFF), 5)

        total_passed_vehicle = total_passed_vehicle + counter

        # insert information text to video frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            input_frame,
            'Veiculos Detectados: ' + str(total_passed_vehicle),
            (10, 35),
            font,
            0.8,
            (0, 0xFF, 0xFF),
            2,
            cv2.FONT_HERSHEY_SIMPLEX,
            )               
        
        cv2.putText(
            input_frame,
            'Linha de ROI',
            (545, roi-10),
            font,
            0.6,
            (0, 0, 0xFF),
            2,
            cv2.LINE_AA,
            )
        # image = utils.draw_bbox(frame, bboxes)
        # curr_time = time.time()
        # exec_time = curr_time - prev_time
        # result = np.asarray(image)
        # info = "time: %.2f ms" %(1000*exec_time)
        # print(info)
        # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        # result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # cv2.imshow("result", result)
        # if cv2.waitKey(1) & 0xFF == ord('q'): break
        output_movie.write(frame)
        print ("writing frame")

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    vid.release()   
    output_movie.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
