# %%
import cv2
import numpy as np
from sklearn.metrics import pairwise

# %%
background = None

accumulated_weight = 0.5

# red square where you put your hand in
roi_top = 10
roi_bottom = 300
roi_rigt = 300
roi_left = 650

# %%
# find the avarage background value
def calculate_accumulate_avarage(frame,accumulated_weight):
    global background
    if background is None:
        background = frame.copy().astype('float')
        return None
    
    cv2.accumulateWeighted(frame,background,accumulated_weight)

# %%
def segment(frame,threshold=25):
    diff = cv2.absdiff(background.astype('uint8'),frame) # absolute difference between background and frame
    
    ret,thresholded=cv2.threshold(diff,threshold,255,cv2.THRESH_BINARY) # this function calcas the threshold of the diff, we pass
                                                                        # threshold as min, the 255 as max and the type of calculation
                                                                        # we want. 
    
    contours, hierarchy = cv2.findContours(thresholded.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE) # heare we find the contours of
                                                                                                         # the hand
    
    if len(contours) == 0: # If the hand contours are not found, we return none.
        return None

    else: # we return the hand segment. that is the max value of the list of contours.
        hand_segment = max(contours,key=cv2.contourArea)
        
        return (thresholded,hand_segment)

# %%
def count_fingers(thresholded,hand_segment):
    conv_hull = cv2.convexHull(hand_segment) # Conv_hull draws a polygon by connecting points around
                                             # the most exernal points in a frame.
    
    top = tuple(conv_hull[conv_hull[:,:,1].argmin()][0]) # most extreme point up
    bottom = tuple(conv_hull[conv_hull[:,:,1].argmax()][0]) # most extreme point down
    left = tuple(conv_hull[conv_hull[:,:,0].argmin()][0]) # most extreme point left
    right = tuple(conv_hull[conv_hull[:,:,0].argmax()][0]) # most extreme point right
    
    center_x = (left[0]+right[0])//2 # calc the center of the line between left and right points
    center_y = (top[1]+bottom[1])//2 # calc the center of the line between top and bottom points
    
    distance = pairwise.euclidean_distances([(center_x, center_y)],Y=[left,right,top,bottom])[0] # we calc the extreme distance to the
                                                                                                 # center.
    max_distance = distance.max() 
    
    radius = int(0.7*max_distance) # We use that distance to choose the radius of the circle. What's outside the circle 
                                    # is the fingers up.
    circumfrence = (2*np.pi*radius)
    
    circular_roi = np.zeros(thresholded.shape[:2],dtype='uint8')
    
    cv2.circle(circular_roi, (center_x, center_y), radius,255,10)
    
    circular_roi = cv2.bitwise_and(thresholded,thresholded,mask=circular_roi)
    countors,hierarchy = cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    f_count = 0
    
    for cnt in countors: # function to count the fingers (anything that is off the circular_roi)
        (x,y,w,h) = cv2.boundingRect(cnt)
        
        out_of_wrist = (center_y+(center_y*0.25))>(y+h)
        
        limit_points = ((circumfrence*0.25)>cnt.shape[0])
        
        if out_of_wrist and limit_points:
            f_count += 1
            
    return f_count

# %%
cam = cv2.VideoCapture(0)

number_of_frames = 0

while True:
    ret, frame = cam.read()
    frame_copy = frame.copy()
    
    roi = frame[roi_top:roi_bottom,roi_rigt:roi_left]
    
    gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(7,7),0)
    
    if number_of_frames < 60:
        calculate_accumulate_avarage(gray,accumulated_weight)
        
        if number_of_frames<=59:
            cv2.putText(frame_copy,'Getting Background', (200,300),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            cv2.imshow('finger count', frame_copy)
    else:
        hand = segment(gray)
        
        if hand is not None:
            threshholded, hand_segment = hand
            
            cv2.drawContours(frame_copy,[hand_segment+(roi_rigt,roi_top)],-1,(255,0,0),5)
            
            fingers = count_fingers(threshholded, hand_segment)
            
            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            
            cv2.imshow('Threshold',threshholded)
            
    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_rigt,roi_bottom),(0,0,255),5)
    
    number_of_frames+=1
    
    cv2.imshow('Finger Count',frame_copy)
    
    k = cv2.waitKey(1) & 0xFF
    
    if k==27:
        break
    
cam.release()
cv2.destroyAllWindows()


