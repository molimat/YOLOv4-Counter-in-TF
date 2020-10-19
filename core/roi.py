import cv2
import numpy as np

refPt = ([])
image = []
clone = []
cropping = False

def click_and_draw(event, x, y, flags, param):
    # grab references to the global variables
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    global refPt, cropping, image, clone

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping == True:
            image = clone.copy()
            if refPt[0][0] != x & refPt[0][1] != y:
                cv2.line(image, refPt[0], (x, y), (255, 0, 0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        cropping = False

def get_ROI_line(img):
    # # construct the argument parser and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--image", required=True, help="Path to the image")
    # args = vars(ap.parse_args())
    # # load the image, clone it, and setup the mouse callback function
    global refPt, cropping, image, clone
    image = np.array(img)
    image = cv2.putText(image, "Press C to confirm ROI and continue.", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA )
    image = cv2.putText(image, "Press R to reset.", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA )
    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_draw)
    

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            cv2.destroyAllWindows()
            break
    # if there are two reference points, then crop the region of interest
    # from teh image and display it
    if len(refPt) == 2:
        cropping = False
        cv2.line(image, refPt[0], refPt[1], (0, 255, 255), 3)
        cv2.waitKey(0)
    # close all open windows
    cv2.destroyAllWindows()
    roi_line = refPt
 
    return roi_line