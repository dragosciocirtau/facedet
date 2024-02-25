import cv2
import numpy as np
import win32api, win32con
from datetime import datetime, timedelta
from time import sleep


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)
display_width = win32api.GetSystemMetrics(0)
display_height = win32api.GetSystemMetrics(1)
corners = {1: "upper left", 2: "lower left", 3: "lower right", 4: "upper right"}
display_corners = []
used_eye = 0


def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        frame = img[y:y + h, x:x + w]
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,225,255),2)
        eyecenter = x + w / 2  # get the eye center
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
        else:
            right_eye = img[y:y + h, x:x + w]
    return left_eye, right_eye


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

    return img


def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    keypoints = detector.detect(img)
    print(keypoints)
    return keypoints


def calculate_brightness(image):
    if image is None:
        return 0
    image_size_half = len(image[0]) // 2
    brightness = np.mean(image[:image_size_half][:])

    return brightness / 2


def eye_possition(img, keypoints):
    if keypoints != (): 
        keypoint = keypoints[0]
        eye_x, eye_y = 0,0
        if keypoint != None:
            eye_x = keypoint.pt[0]
            eye_y = keypoint.pt[1]
        
        print(eye_x, eye_y)
        eye_x = eye_x / len(img) * display_width
        eye_y = eye_y / len(img[0]) * display_height
        print(eye_x, eye_y)
        
        move_cursor(int(eye_x), int(eye_y))


def detect_next_corner(corner_number):
    print("keep your head still and facing the display and move your eyes towards the " + corners[corner_number + 1] + " corner of the screen for the next 3 seconds")
    sleep(3)
    

def get_average_point_for_corner(current_corner_pos):
    eye_x = 0
    eye_y = 0
    for point in current_corner_pos:
        eye_x += point.pt[0]
        eye_y += point.pt[1]
    
    display_corners.append((eye_x / len(current_corner_pos), eye_y / len(current_corner_pos)))


def move_cursor(x,y):
    win32api.SetCursorPos((x,y))

def main():
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    end_time = datetime.now()
    current_corner_pos = []
    corners_detected = 0
    
    while True:
        _, frame = cap.read()
        face_frame = detect_faces(frame, face_cascade)
        threshold = calculate_brightness(face_frame)
        
        if (corners_detected < 4 and end_time < datetime.now()):
            
            if len(current_corner_pos) != 0:
                get_average_point_for_corner(current_corner_pos)
                corners_detected = corners_detected + 1
                
            if corners_detected < 4: 
                current_corner_pos = []
                detect_next_corner(corners_detected)
                end_time = datetime.now() + timedelta(seconds=3)
        
        if corners_detected == 4:
            print(display_corners)
            break
            
        if face_frame is not None:
            eyes = detect_eyes(face_frame, eye_cascade)
            for eye in eyes:
                if eye is not None:
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    if keypoints != ():
                        current_corner_pos.append(keypoints[used_eye])
                    #eye_possition(eye, keypoints)
                    cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            #eye_possition(eyes[0], keypoints)
        cv2.imshow('image', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()