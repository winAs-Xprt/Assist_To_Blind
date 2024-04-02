import math
import pyttsx3
from ultralytics import YOLO
import cv2
import cvzone

# Initialize the text-to-speech engine
engine = pyttsx3.init()

cap=cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)

model=YOLO("../YOLO-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while True:
    success,img=cap.read()
    results=model(img,stream=True)
    for r in results:
        boxes=r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1, y1, x2, y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            w,h=x2-x1,y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            conf=math.ceil((box.conf[0]*100))/100
            print(conf)

            # class name
            cls=int(box.cls[0])
            class_name = classNames[cls]

            # concatenate location information to output string
            center_x = (x1 + x2) // 2  # adding the value of x1 and x2 in the particular detected object that is the axes
            img_width = img.shape[1]   # width of the detected image
            location = " on left move right" if center_x < img_width // 2 else "on right move left"
            cls_name=f'{class_name}'

            output_str = f'{class_name}  detected  {location} )'

            cvzone.putTextRect(img, cls_name, (max(0, x1), max(35, y1)))
            # Speak out the class name
            engine.say(output_str)
            engine.runAndWait()

    cv2.imshow("Image",img)
    cv2.waitKey(1)
# Pills info - detecting pills and telling their info through voice