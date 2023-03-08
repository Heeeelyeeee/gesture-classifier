import cv2
import mediapipe as mp
  
#############
wCam = 320*2
hCam = 240*2
#############


# define a video capture object
vid = cv2.VideoCapture(0,cv2.CAP_DSHOW)
vid.set(3,wCam)
vid.set(4,hCam)
mpHands = mp.solutions.hands
hands = mpHands.Hands() #can alter here to add more hands
mpDraw = mp.solutions.drawing_utils
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRGB)

    if results.multi_hand_landmarks:
        for handlms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(frame, handlms, mpHands.HAND_CONNECTIONS)
    
    # Display the resulting frame
    cv2.imshow('Hangesture With openCV', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()