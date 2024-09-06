import math
from ultralytics import YOLO
import cv2
import os
os.chdir(r"F:\YOLO Projects\HumanActivity")

# Function to calculate the angle between three points
def calculate_angle(pointA, pointB, pointC):
    vectorAB = [pointA[0] - pointB[0], pointA[1] - pointB[1]]
    vectorCB = [pointC[0] - pointB[0], pointC[1] - pointB[1]]

    
    dot_product = vectorAB[0] * vectorCB[0] + vectorAB[1] * vectorCB[1]
    magnitudeAB = math.sqrt(vectorAB[0] ** 2 + vectorAB[1] ** 2)
    magnitudeCB = math.sqrt(vectorCB[0] ** 2 + vectorCB[1] ** 2)

    
    angle_rad = math.acos(dot_product / (magnitudeAB * magnitudeCB + 1e-6)) 
    angle_deg = math.degrees(angle_rad) 
    return angle_deg

model = YOLO('yolov8s-pose.pt')

cap = cv2.VideoCapture('cc (1).mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    frame_resized = cv2.resize(frame, (1100, 700))

    
    results = model(frame_resized, conf=0.8)

    for result in results:
        
        boxes = result.boxes.xyxy  
        keypoints = result.keypoints.xy  
        
        
        for i, box in enumerate(boxes):
            
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  

        
            left_shoulder = keypoints[i][5]
            left_hip = keypoints[i][11]
            left_knee = keypoints[i][13]

            
            cv2.circle(frame_resized, (int(left_shoulder[0]), int(left_shoulder[1])), 5, (0, 0, 255), -1)  # Red circle for shoulder
            cv2.circle(frame_resized, (int(left_hip[0]), int(left_hip[1])), 5, (0, 0, 255), -1)            # Red circle for hip
            cv2.circle(frame_resized, (int(left_knee[0]), int(left_knee[1])), 5, (0, 0, 255), -1)          # Red circle for knee
            
           
            cv2.line(frame_resized, (int(left_shoulder[0]), int(left_shoulder[1])), (int(left_hip[0]), int(left_hip[1])), (255, 0, 0), 2)
            cv2.line(frame_resized, (int(left_hip[0]), int(left_hip[1])), (int(left_knee[0]), int(left_knee[1])), (255, 0, 0), 2)

            
            angle = calculate_angle(left_shoulder, left_hip, left_knee)

            
            if angle < 140:  
                activity = "Sitting"
            else:  
                activity = "Standing"

           
            cv2.putText(frame_resized, f'Activity: {activity}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) , 2)
            cv2.putText(frame_resized, f'Angle: {int(angle)}', (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    
    cv2.imshow('Pose Detection with Activity', frame_resized)

    if cv2.waitKey(1) & 0xFF == ord('x'):
        break


cap.release()
cv2.destroyAllWindows()
