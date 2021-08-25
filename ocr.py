import cv2
import numpy as np
import time
import os



# read class names from text file
classes = ["1","9","6","0","2","3","4","5","8","7"]

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# read pre-trained model and config file
net = cv2.dnn.readNet("ocrmodel.weights", "ocrmodel.cfg")
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def ocr(image):

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)

    # function to get the output layer names 
    # in the architecture

    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))

    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                
        

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # go through the detections remaining
    # after nms and draw bounding box
    xlist=[]
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        # draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))
        # L.append(classes[class_ids[i]])
        # cv2.imshow("ocr",image)
        # cv2.waitKey(0)
        xlist.append((x,class_ids[i],confidences[i]))
    xlist.sort(key=lambda tup: tup[0])
    ocrlist=[]
    scores=[]
    for j in xlist:
        ocrlist.append(classes[j[1]])
        scores.append((classes[j[1]],j[2]))
    ocr=''.join(ocrlist)
    # release resources
    # cv2.putText(image, ''.join(ocr), (5,7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,0,0), 1)
    try:
        return ocr,scores
    except:
        pass





    

