import cv2
import numpy as np

def faceBox(faceNet, frame, conf_threshold=0.8):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    bboxs = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            bboxs.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return frame, bboxs

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = [
    '(0-2)', '(3-4)', '(5-7)', '(8-12)', '(13-15)',
    '(16-20)', '(21-24)', '(25-32)', '(33-37)', '(38-43)',
    '(44-47)', '(48-53)', '(54-59)', '(60-100)'
]
genderList = ['Male', 'Female']

# Start video capture
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    
    frame, bboxs = faceBox(faceNet, frame)
    for bbox in bboxs:
        face = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        
        if face.size == 0:
            continue
        
        # Resize the face for prediction
        face_resized = cv2.resize(face, (227, 227), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(face_resized, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=True)

        # Predict gender
        genderNet.setInput(blob)
        genderPred = genderNet.forward()
        gender = genderList[np.argmax(genderPred)]

        # Predict age
        ageNet.setInput(blob)
        agePred = ageNet.forward()
        age = ageList[np.argmax(agePred)]

        label = "{}, {}".format(gender, age)
        cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imshow("Age-Gender Detection", frame)
    
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
