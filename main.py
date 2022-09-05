import cv2
from gui_buttons import Buttons

button = Buttons()
button.add_button("person", 20, 20)
button.add_button("keyboard", 20, 80)
 
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights", "dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)
 
classes = []
with open("dnn_model/classes.txt", "r") as file_object :
  for class_name in file_object.readlines():
    class_name = class_name.strip()
    classes.append(class_name)
 
cap = cv2.VideoCapture("sample.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
 
def click_button(event, x, y, flags, params) :
  global button_parson
  if event == cv2.EVENT_LBUTTONDOWN:
    button.button_click(x, y)

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)
 
while True :
  ret, frame = cap.read()
  
  active_buttons = button.active_buttons_list()
 
  (class_ids, score, bboxes) = model.detect(frame)
  for (class_id, score, bbox) in zip(class_ids, score, bboxes):
    (x, y, h, w) =  bbox
   
    class_name = classes[class_id]
   
    if class_name in active_buttons:
        cv2.putText(frame, str(class_name), (x, y -10), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 0, 500), 2)
        cv2.rectangle(frame, (x, y), (x + w , y + h), (200, 0, 50), 3)

    button.display_buttons(frame)
    
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF==ord('q') :
        break

