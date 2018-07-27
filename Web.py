import cv2
import numpy as np

def main(model,dummy_array,decode_netout,NMS_THRESHOLD,ANCHORS,CLASS,LABELS,draw_boxes):
    cam = cv2.VideoCapture(0)

    cam.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 100)
    cam.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 100)
    while (True):
        ret, image = cam.read()
        # image preprocessing
        input_image = cv2.resize(image, (416, 416))
        input_image = input_image / 255.
        input_image = input_image[:, :, ::-1]
        input_image = np.expand_dims(input_image, 0)

        netout = model.predict([input_image, dummy_array])

        boxes = decode_netout(netout[0],
                              obj_threshold=0.3,
                              nms_threshold=NMS_THRESHOLD,
                              anchors=ANCHORS,
                              nb_class=CLASS)

        image = draw_boxes(image, boxes, labels=LABELS)
        cv2.imshow('frame', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

