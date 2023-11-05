import cv2
import numpy as np


#################################################



###############I have use  yolo pretrinned  files    to read and apply processing to detct the person and also count it

##################################################
photo = cv2.imread('download.jpg')   ###  Add your image path here 
model = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')    ###pretrained yolo files uploaded 
output_layers = model.getUnconnectedOutLayersNames()  ######  getting output layers 
(H, W) = photo.shape[:2]    ##########Extracting the image properties (lenght and width)
blob = cv2.dnn.blobFromImage(photo, 1 / 255.0, (416, 416), swapRB=True, crop=False)  ### blob function for input image
model.setInput(blob)   ###########set model with blob
outputs = model.forward(output_layers);    ############output
boxes = []     ##############Define the boxes
confidences = []     ############confidence 0r score declaration
for output in outputs:                      #######using for loop to get each output from output
    for detection in output:
        scores = detection[5:]      ############score location
        class_id = np.argmax(scores)     ##########Getting class ids 
        confidence = scores[class_id]        ############conficence

        # Filter out weak detections
        if confidence > 0.3:    #############Now threhold 0.3 declaration you can set the calue up to 1 
            # Get the bounding box coordinates for the detected object
            box = detection[0:4] * np.array([W, H, W, H])        ###############After detection get boxes 
            (centerX, centerY, width, height) = box.astype('int')

            # Calculate the top-left and bottom-right coordinates of the bounding box
            x = int(centerX - (width / 2))              ##############   find out points initial in image top left and bottom right 
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
# Perform non-maximum suppression to suppress weak, overlapping detections
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)   ##########Remove double boxes in a single image 
###############################################################################
count = 0     ###########  counting algothm 
if(len(indices) > 0):     
    for i in indices.flatten():
        label = 'person'    #############Get only person classs in label 
        if label == 'person':
            count += 1    ##############add one by by by see it image how many person in a image 
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
       # text = ' {:.4f}'.format(confidences[i])
     #   cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.rectangle(photo, (x, y), (x + w, y + h), (230,55,255), 2)
cv2.putText(photo,("Total Count="+str(count)),(15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
cv2.imshow("image",photo)
cv2.waitKey(1)
print(f'People Detected ={count}')

