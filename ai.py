import mediapipe as mp
import time
import cv2
from ultralytics import YOLO
from collections import defaultdict

def finger_count():
    video = cv2.VideoCapture(0)

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands = 1)
    mpDraw = mp.solutions.drawing_utils

    tip = [4, 8, 12, 16, 20]
    while True:
        sucess, img = video.read()
        img = cv2.flip(img, 1)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        lmlist = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                for id, lm in enumerate(handLms.landmark):
                    h,w,c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmlist.append((id, cx, cy))

                mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        # print(lmlist)
        if(len(lmlist)):
            finger = []            
            for id in range(1, 5):
                if lmlist[tip[id]][2] < lmlist[tip[id] - 2][2]:
                    finger.append(1)
                else:
                    finger.append(0)

            totalFingers = sum(finger)

            if(totalFingers == 2):
                count_cars()

            cv2.putText(img, str(totalFingers), (575,45), cv2.FONT_HERSHEY_PLAIN,3,(0,255,0), 3)


        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

def count_cars():
    line_start_x = 200
    line_start_y = 100
    line_end_x = 600
    line_end_y = 100

    model = YOLO('yolov8x.pt')

    cap = cv2.VideoCapture("Cars.mp4")

    track_history=defaultdict(lambda:[])
    crossed_track_ids = set()
    crossed_objects_count = 0

    class_names = [
        'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'sofa', 'potted plant', 'bed', 'dining table', 'toilet', 'tvmonitor', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    while cap.isOpened():
        success,frame = cap.read()
        if not success:
            break

        results = model.track(frame, classes = [2,5,7], persist = True, save = True, tracker = "bytetrack.yaml")

        for box,track_id,cls in zip(results[0].boxes.xywh, results[0].boxes.id.int().tolist(), results[0].boxes.cls.int().tolist()):
            x,y,w,h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))
            if len(track)>30:
                track.pop(0)

            if (track and len(track)>1 and (track[0][0]<line_start_x and track[-1][0]>line_end_x) and track_id not in crossed_track_ids):
                crossed_objects_count += 1
                crossed_track_ids.add(track_id)

            cv2.rectangle(frame, (int(x-w/2), int(y-h/2)),( int(x+w/2), int(y+h/2)), (0,255,0), 4)
            class_name = class_names[cls]

            text_x = int(x-w/2)+5
            text_y = int(y-h/2)-15
            cv2.putText(frame, class_name,(text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,255,0),3)

        resized_frame = cv2.resize(frame,None, fx = 0.9, fy = 0.9)
        cv2.namedWindow('Object Detection and Tracking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Detection and Tracking', 800, 600)

        cv2.imshow('Object Detection and Tracking', resized_frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
        cv2.putText(frame, str(crossed_objects_count),(100,100), cv2.FONT_HERSHEY_SIMPLEX,3.0,(0,0,0),5)

        if 8 in track_history:
            print("Overcrowded")
            break

    cap.release()
    cv2.destroyAllWindows() 


finger_count()