import cv2
from mediapipe.framework.formats import landmark_pb2

from LandMarkerandResults import landmarker_and_result
import mediapipe as mp
import numpy as np
# image = cv2.imread("") # Reading a file
# cv2.imshow("Local", image) # Displays the image
# cv2.imwrite("My-logo.jpeg", image) # Saves image in a different format.
#
# cv2.waitKey() # Waits until a key is pressed until it closes
#
# cv2.destroyAllWindows() # Destroys all window

# model = '/Users/jonathanmeyers/PycharmProjects/HandTracking/hand_landmarker.task' # Model to use
# BaseOptions = mp.tasks.BaseOptions
# HandLandMarker = mp.tasks.vision.HandLandmarker
# HandLandMarkerOptions = mp.tasks.vision.HandLandmarkerOptions
# HandLandmarkerResults = mp.tasks.vision.HandLandmarkerResult
# VisionRunningMode = mp.tasks.vision.RunningMode

cap = cv2.VideoCapture(0) # Setting camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)

hand_landmarker = landmarker_and_result()


def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
    """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
    try:
        if detection_result.hand_landmarks == []:
            return rgb_image
        else:
            hand_landmarks_list = detection_result.hand_landmarks
            annotated_image = np.copy(rgb_image)

            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
                    hand_landmarks])
                mp.solutions.drawing_utils.draw_landmarks(
                    annotated_image,
                    hand_landmarks_proto,
                    mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style())
            return annotated_image
    except:
        return rgb_image

# mp_hands = mp.solutions.hands # mp_hands is an actual module
# hand = mp_hands.Hands()

def count_fingers_raised(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Iterate through each hand, checking if fingers (and thumb) are raised.
   Hand landmark enumeration (and weird naming convention) comes from
   https://developers.google.com/mediapipe/solutions/vision/hand_landmarker."""
   try:
      # Get Data
      hand_landmarks_list = detection_result.hand_landmarks

      # Code to count numbers of fingers raised will go here
      numRaised = 0
     # for each hand
      for idx in range(len(hand_landmarks_list)):
          hand_landmarks = hand_landmarks_list[idx]
          # For each fingertip
          for i in range(8,21,4):
              tip_y = hand_landmarks[i].y
              dip_y = hand_landmarks[i-1].y
              pip_y = hand_landmarks[i-2].y
              mcp_y = hand_landmarks[i-3].y
              if tip_y < min(dip_y, pip_y, mcp_y):
                  numRaised += 1
            # For the thumb
          tip_x = hand_landmarks[4].x
          dip_x = hand_landmarks[3].x
          pip_x = hand_landmarks[2].x
          mcp_x = hand_landmarks[1].x
          palm_x = hand_landmarks[0].x
          if mcp_x > palm_x:
              if tip_x > max(dip_x, pip_x,mcp_x):
                  numRaised +=1
              else:
                  if tip_x < min(dip_x, pip_x, mcp_x):
                      numRaised += 1


      # Code to display the number of fingers raised will go here
      annotated_image = np.copy(rgb_image)
      height, width, _ = annotated_image.shape
      text_x = int(hand_landmarks[0].x * width) - 100
      text_y = int(hand_landmarks[0].y * height) + 50
      cv2.putText( img=annotated_image, text=str(numRaised) + " Fingers Raised", org=(text_x, text_y), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                   fontScale=1, color=(0,0,255), thickness=2, lineType=cv2.LINE_4)

      return annotated_image
   except:
      return rgb_image

while True:
    ret, frame = cap.read()  # Returns image from camera
    #frame = cv2.flip(frame, -1)
    if ret:
        frame = draw_landmarks_on_image(frame, hand_landmarker.result)
        frame = count_fingers_raised(frame, hand_landmarker.result)
        cv2.imshow("Capture Image", frame)
        hand_landmarker.detect_async(frame)
        print(hand_landmarker.result)
        #frames = draw_landmarks_on_image(frame, hand_landmarker.result)
        #mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame) # Convert frame received from openCO to a image object.
        if cv2.waitKey(1) == ord('q'):
            break

hand_landmarker.close()
cv2.destroyAllWindows()