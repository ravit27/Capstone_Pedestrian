from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog
import os

# import the necessary packages
import numpy as np
import imutils
import argparse
import time
import cv2
import os
import PySimpleGUI as sg


def openwebcam():
        y_path = r'yolo-coco'

        sg.theme('LightGreen')

        gui_confidence = .5     # initial settings
        gui_threshold = .3      # initial settings
        camera_number = 0       # if you have more than 1 camera, change this variable to choose which is used

        # load the COCO class labels our YOLO model was trained on
        labelsPath = r"F:/PersonDetectionAndCounting/coco.names"
        LABELS = open(labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

        # derive the paths to the YOLO weights and model configuration
        weightsPath = r"F:/PersonDetectionAndCounting/yolov3.weights"
        configPath = r"F:/PersonDetectionAndCounting/yolov3.cfg"

        # load our YOLO object detector trained on COCO dataset (80 classes)
        # and determine only the *output* layer names that we need from YOLO
        sg.popup_quick_message('Loading YOLO weights from disk.... one moment...', background_color='red', text_color='white')

        net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

        # initialize the video stream, pointer to output video file, and
        # frame dimensions
        W, H = None, None
        win_started = False
        cap = cv2.VideoCapture(camera_number)  # initialize the capture device
        while True:
                # read the next frame from the file or webcam
                grabbed, frame = cap.read()

                # if the frame was not grabbed, then we stream has stopped so break out
                if not grabbed:
                        break

                # if the frame dimensions are empty, grab them
                if not W or not H:
                        (H, W) = frame.shape[:2]

                # construct a blob from the input frame and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes
                # and associated probabilities
                blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
                net.setInput(blob)
                start = time.time()
                layerOutputs = net.forward(ln)
                end = time.time()

                # initialize our lists of detected bounding boxes, confidences,
                # and class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []

                # loop over each of the layer outputs
                for output in layerOutputs:
                        # loop over each of the detections
                        for detection in output:
                                # extract the class ID and confidence (i.e., probability)
                                # of the current object detection
                                scores = detection[5:]
                                classID = np.argmax(scores)
                                confidence = scores[classID]

                                # filter out weak predictions by ensuring the detected
                                # probability is greater than the minimum probability
                                if confidence > gui_confidence:
                                        # scale the bounding box coordinates back relative to
                                        # the size of the image, keeping in mind that YOLO
                                        # actually returns the center (x, y)-coordinates of
                                        # the bounding box followed by the boxes' width and
                                        # height
                                        box = detection[0:4] * np.array([W, H, W, H])
                                        (centerX, centerY, width, height) = box.astype("int")

                                        # use the center (x, y)-coordinates to derive the top
                                        # and and left corner of the bounding box
                                        x = int(centerX - (width / 2))
                                        y = int(centerY - (height / 2))

                                        # update our list of bounding box coordinates,
                                        # confidences, and class IDs
                                        boxes.append([x, y, int(width), int(height)])
                                        confidences.append(float(confidence))
                                        classIDs.append(classID)

                # apply non-maxima suppression to suppress weak, overlapping bounding boxes
                idxs = cv2.dnn.NMSBoxes(boxes, confidences, gui_confidence, gui_threshold)

                count=0
                color2=(0,0,0)

                # ensure at least one detection exists
                if len(idxs) > 0:
                        # loop over the indexes we are keeping
                        for i in idxs.flatten():
                                if(  LABELS[classIDs[i]])=='person':
                                        # extract the bounding box coordinates
                                        (x, y) = (boxes[i][0], boxes[i][1])
                                        (w, h) = (boxes[i][2], boxes[i][3])
                                        # draw a bounding box rectangle and label on the frame
                                        color = [int(c) for c in COLORS[classIDs[i]]]
                                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                                        text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                               confidences[i])
                                        cv2.putText(frame, text, (x, y - 5),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                        
                                        count=count+1
                                        cv2.rectangle(frame,(0,0),(250,17),color2,20)
                                        
                                        cv2.putText(frame,"No Of Person: "+(str)(count), (0, 22),  
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                if(count==0):
                                        cv2.rectangle(frame,(0,0),(250,17),color2,20)
                                        
                                        cv2.putText(frame,"No Of Person: "+(str)(count), (0, 22),  
                                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                else:
                        cv2.rectangle(frame,(0,0),(250,17),color2,20)
                        cv2.putText(frame,"No Of Person: 0",(0, 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                                        
                                        

                imgbytes = cv2.imencode('.png', frame)[1].tobytes()
    # ---------------------------- THE GUI ----------------------------
                if not win_started:
                        win_started = True
                        
                        layout = [
                            [sg.Text("Object Detection", size=(30, 1))],
                            [sg.Image(data=imgbytes, key='_IMAGE_')],
                            [sg.Text('Confidence'),
                             sg.Slider(range=(0, 10), orientation='h', resolution=1, default_value=5, size=(15, 15), key='confidence'),
                             sg.Text('Threshold'),
                             sg.Slider(range=(0, 10), orientation='h', resolution=1, default_value=3, size=(15, 15), key='threshold')],
                            [sg.Exit()]
                        ]
                        win = sg.Window('YOLO Webcam Demo', layout, default_element_size=(14, 1), text_justification='right', auto_size_text=False, finalize=True)
                        image_elem = win['_IMAGE_']
                else:
                        image_elem.update(data=imgbytes)

                event, values = win.read(timeout=0)
                if event is None or event == 'Exit':
                        break
                gui_confidence = int(values['confidence']) / 10
                gui_threshold = int(values['threshold']) / 10

        print("[INFO] cleaning up...")
        win.close()
        
        
def exit():
        root.destroy()
        

def refresh():
        bottomframe.destroy()
        
        
root = Tk()
root.title("Pedestrian Detection")
root.geometry("800x600")
root.resizable(width=True, height=True)
root.config(bg='gray78')




lbl = Label(root, 
                text="Pedestrian Detection",
                compound = CENTER,
                padx = 10,
                fg = "midnight blue",
                bg = "gainsboro",
                font = "Georgia 30 bold "
                ).pack(ipadx=10,        ipady=10,       padx=30,        pady=10, fill = X)


frame = Frame(root, width=200,height=200, bg='gray78')
frame.pack()



btn = Button(frame, text='Open webcam',
                fg = "midnight blue",
                bg = "lavender",
                font = "Helvetica 16 bold",
                command=openwebcam).pack(
                side=LEFT,
                ipadx=2,
                ipady=2,
                padx=10,
                pady=10)
        
refresh_btn = Button(frame, text='Refresh',
                fg = "green",
                bg = "lavender",
                font = "Helvetica 16 bold",
                command=refresh).pack(
                side=LEFT,
                ipadx=2,
                ipady=2,
                padx=10,
                pady=10)        
exit_btn = Button(frame, text='Exit',
                fg = "red",
                bg = "lavender",
                font = "Helvetica 16 bold",
                command=exit).pack(
                side=RIGHT,
                ipadx=2,
                ipady=2,
                padx=10,
                pady=10)
                

bottomframe = Frame(root,width=800,height=400, bg='gray78')
                
root.mainloop()