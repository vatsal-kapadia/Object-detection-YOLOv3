import cv2
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import sys
# %matplotlib inline

weights = r"/Users/vatsalkapadia/Downloads/Neural Network/yolov3.weights"
yolo_cfg = r"/Users/vatsalkapadia/Downloads/Neural Network/yolov3.cfg"



j = 0
def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (int(x),int(y)), (int(x_plus_w),int(y_plus_h)), color, 2)
    cv2.putText(img, label, (int(x)-10,int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
   


cap = cv2.VideoCapture(r"/Users/vatsalkapadia/Desktop/Simple_Vehicle_Detection/video.avi")
cv2.namedWindow("object detected ",cv2.WINDOW_NORMAL)

   

while(cap.isOpened()):
     

  ret, frame = cap.read()
  if ret == True:  
    image = frame
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    classes = None
   
    with open(r"/Users/vatsalkapadia/Downloads/Neural Network/yolov3.txt") as f:
        classes = [line.strip() for line in f.readlines()]
       
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    net = cv2.dnn.readNet(weights, yolo_cfg)
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
   
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
   
    for out in outs:
     for detection in out:
         scores = detection[5:]
         
         class_id = np.argmax(scores)
         confidence = scores[class_id]
         if confidence > 0.3:
             print(confidence)
             center_x = int(detection[0] * Width)
             center_y = int(detection[1] * Height)
             w = int(detection[2] * Width)
             h = int(detection[3] * Height)
             x = center_x - w / 2
             y = center_y - h / 2
             class_ids.append(class_id)
             confidences.append(float(confidence))
             boxes.append([x, y, w, h])
   
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
   
    for i in indices:
     i = i[0]
     box = boxes[i]
     x = box[0]
     y = box[1]
     w = box[2]
     h = box[3]
     draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
    count = str(len(indices))
    cv2.imshow("object detected ", image)
   
    if cv2.waitKey(27) & 0xFF == ord('q'):
      break
   
  else:
    break



cv2.destroyAllWindows()








    


