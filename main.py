
#v8
import cv2
import numpy as np
import math
import skfuzzy  as fuzz
from skfuzzy import control as ctrl
import skfuzzy.membership as mf
from alarm import play_sound

cap = cv2.VideoCapture(0)
fps = int(cap.get(cv2.CAP_PROP_FPS))
sec = 5
threshold = 20
kernel = np.ones((9, 9))/81
frame_count = 0

def get_distance(center, x, y, w, h):
    # distance = abs(center[0]-(x+w/2))+(center[1]-(y+h/2))
    distance = math.sqrt((center[0] - (x + w/2))**2 + (center[1] - (y + h/2))**2)
    # print(int(distance))
    return distance 

def get_gray_level_difference(frame, background, x, y, w, h):
    current_region = frame[y:y+h, x:x+w]
    background_region = background[y:y+h, x:x+w]
    
    difference_regeion = np.abs(current_region - background_region)
    avg_region =  np.mean(difference_regeion)
    
    return avg_region
    
    
def get_size(w, h):
    return w*h

def alarm():
    path = "sound1.wav"
    play_sound(path)

    
def get_background():
    background_frames = []  
    for i in range(sec * fps):
        ret, frame = cap.read()
        background_frames.append(frame)
        if not ret:
            print("Failed to read frame")
            exit()

    # Initialize background image and grayscale background
    background = np.average(background_frames, axis=0).astype(np.uint8)
    gray_background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('Gray_background',gray_background)
    gray_background = gray_background.astype(np.float32)
    height, width = gray_background.shape[:2]
    return gray_background, width , height


new_background , width , height= get_background()
# print(f"witdh:{width},h:{height} "  )
center = (width/2 , height/2)

# # fuzzy inputs algorithm
# max_distance = math.sqrt((width/2)**2 + (height/2)**2)
# print(f"max_distance: {max_distance}")
# distance_range = np.arange(-1, max_distance + 1, 1)
# distance = ctrl.Antecedent(distance_range, 'distance')
new_max_distance = 1000  # Adjust this value based on your requirements
# distance = ctrl.Antecedent(np.arange(-1, new_max_distance + 1, 1), 'distance')
difference = ctrl.Antecedent(np.arange(-1, 255, 1), 'difference')
size = ctrl.Antecedent(np.arange(-1, width*height, 1), 'size')
#fuzzy inputs algorithm
attention_level = ctrl.Consequent(np.arange(-1,100,1), "attention_level")
universe = np.arange(-1, 501)
distance = ctrl.Antecedent(universe, 'distance') 


# Define fuzzy sets and membership functions
distance['low'] = mf.trapmf(distance.universe, [-1, 0, 120, 150])
distance['medium'] = mf.trapmf(distance.universe, [120, 150, 230, 266])
distance['high'] = mf.trapmf(distance.universe, [230, 266, 400, 401])

difference['low'] = mf.trapmf(difference.universe, [-1, 0, 75, 100])
difference['medium'] = mf.trapmf(difference.universe, [75, 100, 160, 180])
difference['high'] = mf.trapmf(difference.universe, [160, 180, 255, 256])

size['little'] = mf.trapmf(size.universe, [-1, 0, 35000, 50000])
size['medium'] = mf.trapmf(size.universe, [35000, 50000, 120000, 150000])
size['big'] = mf.trapmf(size.universe, [120000, 150000, 307199, 307200])

attention_level['low'] = mf.trapmf(attention_level.universe, [-1, 0, 30,40])
attention_level['medium'] = mf.trapmf(attention_level.universe, [30, 40, 60, 70])
attention_level['high'] = mf.trapmf(attention_level.universe, [60, 70, 100, 101])

# Rules:  
rule1 = ctrl.Rule(distance['low'] & difference['high'] & size['big'], attention_level['high'])
rule2 = ctrl.Rule(distance['low'] & difference['high'] & size['medium'], attention_level['high'])
rule3 = ctrl.Rule(distance['low'] & difference['high'] & size['little'], attention_level['high'])
rule4 = ctrl.Rule(distance['low'] & difference['medium'] & size['big'], attention_level['high'])
rule5 = ctrl.Rule(distance['low'] & difference['medium'] & size['medium'], attention_level['high'])
rule6 = ctrl.Rule(distance['low'] & difference['medium'] & size['little'], attention_level['medium'])
rule7 = ctrl.Rule(distance['low'] & difference['low'] & size['big'], attention_level['high'])
rule8 = ctrl.Rule(distance['low'] & difference['low'] & size['medium'], attention_level['medium'])
rule9 = ctrl.Rule(distance['low'] & difference['low'] & size['little'], attention_level['medium'])

rule10 = ctrl.Rule(distance['medium'] & difference['high'] & size['big'], attention_level['high'])
rule11 = ctrl.Rule(distance['medium'] & difference['high'] & size['medium'], attention_level['high'])
rule12 = ctrl.Rule(distance['medium'] & difference['high'] & size['little'], attention_level['medium'])
rule13 = ctrl.Rule(distance['medium'] & difference['medium'] & size['big'], attention_level['medium'])
rule14 = ctrl.Rule(distance['medium'] & difference['medium'] & size['medium'], attention_level['medium'])
rule15 = ctrl.Rule(distance['medium'] & difference['medium'] & size['little'], attention_level['low'])
rule16 = ctrl.Rule(distance['medium'] & difference['low'] & size['big'], attention_level['medium'])
rule17 = ctrl.Rule(distance['medium'] & difference['low'] & size['medium'], attention_level['medium'])
rule18 = ctrl.Rule(distance['medium'] & difference['low'] & size['little'], attention_level['low'])

rule19 = ctrl.Rule(distance['high'] & difference['high'] & size['big'], attention_level['medium'])
rule20 = ctrl.Rule(distance['high'] & difference['high'] & size['medium'], attention_level['medium'])
rule21 = ctrl.Rule(distance['high'] & difference['high'] & size['little'], attention_level['low'])
rule22 = ctrl.Rule(distance['high'] & difference['medium'] & size['big'], attention_level['medium'])
rule23 = ctrl.Rule(distance['high'] & difference['medium'] & size['medium'], attention_level['low'])
rule24 = ctrl.Rule(distance['high'] & difference['medium'] & size['little'], attention_level['low'])
rule25 = ctrl.Rule(distance['high'] & difference['low'] & size['big'], attention_level['low'])
rule26 = ctrl.Rule(distance['high'] & difference['low'] & size['medium'], attention_level['low'])
rule27 = ctrl.Rule(distance['high'] & difference['low'] & size['little'] , attention_level['low'])

# Create control system
motion_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9
                                  , rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17
                                  , rule18, rule19, rule20, rule21, rule22, rule23, rule24, rule25,
                                  rule26, rule27])
motion_detection = ctrl.ControlSystemSimulation(motion_ctrl)

while cap.isOpened():
    ret, frame = cap.read()
    
    cv2.imshow('Real_video',frame)
    # Calculate frame difference
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame=gray_frame.astype(np.float32)
    frame_diff =np.abs(gray_frame - new_background)
    
    
    frame_diff_uint8 = frame_diff.astype(np.uint8)
    cv2.imshow('Frame_diff_of_background',frame_diff_uint8)
    
    # Apply threshold 
    thres = cv2.dilate(frame_diff_uint8, kernel, iterations=0)
    ret, thres = cv2.threshold(thres, threshold, 255, cv2.THRESH_BINARY)
    
    #finding contours
    contours, _ = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
    # Calculate area
        area = cv2.contourArea(contour)

    # Calculate perimeter
        # perimeter = cv2.arcLength(contour, True)

    # Get bounding box
        x, y, w, h = cv2.boundingRect(contour)
    filtered_contours = []
    # size_fuzzies = []
    size_fuzzy = 0
    difference_fuzzy = 0
    distance_fuzz = 0
    attention_result = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > (width*height)*0.02:  # Filter by minimum area
            filtered_contours.append(contour)
            distance_fuzz = int(get_distance(center, x, y, w, h))
            # print(distance)
            size_fuzzy=get_size(w,h)
            # size_fuzzies.append(size_fuzzy)
            difference_fuzzy = get_gray_level_difference(gray_frame, new_background, x, y, w, h)
            # motion_detection.input['distance'] = distance
            if distance_fuzz > new_max_distance:
                print("distance_fuzz is more than new_max_distance")
                distance_fuzz = 0
            distance_membership_low = fuzz.interp_membership(distance.universe, distance['low'].mf, distance_fuzz)
            # print(distance_membership_low)
            distance_membership_medium = fuzz.interp_membership(distance.universe, distance['medium'].mf, distance_fuzz)
            distance_membership_high = fuzz.interp_membership(distance.universe, distance['high'].mf, distance_fuzz)
            
            size_membership_little = fuzz.interp_membership(size.universe, size['little'].mf, size_fuzzy)
            size_membership_medium = fuzz.interp_membership(size.universe, size['medium'].mf, size_fuzzy)
            size_membership_big = fuzz.interp_membership(size.universe, size['big'].mf, size_fuzzy)
            
            difference_membership_low = fuzz.interp_membership(difference.universe, difference['low'].mf, difference_fuzzy)
            difference_membership_medium = fuzz.interp_membership(difference.universe, difference['medium'].mf, difference_fuzzy)
            difference_membership_high = fuzz.interp_membership(difference.universe, difference['high'].mf, difference_fuzzy)
            motion_detection.input['distance'] = distance_fuzz
            motion_detection.input['difference'] = difference_fuzzy
            motion_detection.input['size'] = size_fuzzy
            motion_detection.compute()
            attention_result = motion_detection.output['attention_level']
            # print('attention level:', attention_result)
            
            contor_center = x + w/2, y + h/2
            
            # if (attention_result >= 50):
            #     break
        if (attention_result >= 50):              
            break
        
    if (attention_result >= 50):              
            break                   
            
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(thres, (x, y), (x + w, y + h), (255 ,255, 255), 4)
        source_region = new_background[y:y + h, x:x + w]
        gray_frame[y:y + h, x:x + w] = source_region
    # Display binary frame
    cv2.imshow("binary_frame", thres)
    
    #############################################################################
    
    # distance_membership_low = fuzz.interp_membership(distance.universe, distance['low'].mf, distance_fuzz)
    # # print(distance_membership_low)
    # distance_membership_medium = fuzz.interp_membership(distance.universe, distance['medium'].mf, distance_fuzz)
    # distance_membership_high = fuzz.interp_membership(distance.universe, distance['high'].mf, distance_fuzz)
    
    # size_membership_little = fuzz.interp_membership(size.universe, size['little'].mf, size_fuzzy)
    # size_membership_medium = fuzz.interp_membership(size.universe, size['medium'].mf, size_fuzzy)
    # size_membership_big = fuzz.interp_membership(size.universe, size['big'].mf, size_fuzzy)
    
    # difference_membership_low = fuzz.interp_membership(difference.universe, difference['low'].mf, difference_fuzzy)
    # difference_membership_medium = fuzz.interp_membership(difference.universe, difference['medium'].mf, difference_fuzzy)
    # difference_membership_high = fuzz.interp_membership(difference.universe, difference['high'].mf, difference_fuzzy)
   
    ####################################################################################################################
    # Initialize inputs first
    
    # distance_input = {}
    # difference_input = {} 
    # size_input = {}
    
    # distance_input['low'] = distance_membership_low
    # distance_input['medium'] = distance_membership_medium
    # distance_input['high'] = distance_membership_high
    
    # difference_input['low'] = difference_membership_low
    # difference_input['medium'] = difference_membership_medium
    # difference_input['high'] = difference_membership_high
    
    # size_input['little'] = size_membership_little
    # size_input['medium'] = size_membership_medium
    # size_input['big'] = size_membership_big
    
    # input_dict = {}
    # input_dict["distance"] = distance_input 
    # input_dict["difference"] = difference_input
    # input_dict["size"] = size_input

    # motion_detection.input['distance_low'] = distance_membership_low
    # motion_detection.input['distance_medium'] = distance_membership_medium
    # motion_detection.input['distance_high'] = distance_membership_high
    # motion_detection.input['difference_low'] = difference_membership_low
    # motion_detection.input['difference_medium'] = difference_membership_medium
    # motion_detection.input['difference_high'] = difference_membership_high
    # motion_detection.input['size_little'] = size_membership_little
    # motion_detection.input['size_medium'] = size_membership_medium
    # motion_detection.input['size_high'] = size_membership_big
    
    ###############################################################
    
    # motion_detection.input['distance'] = distance_fuzz
    # motion_detection.input['difference'] = difference_fuzzy
    # motion_detection.input['size'] = size_fuzzy
    
    ###############################################################
    # motion_detection.input['distance'] = distance_input
    # motion_detection.input['difference'] = difference_input
    # motion_detection.input['size'] = size_input

    # motion_detection.input['difference'
    
    # motion_detection.input['distance']['low'] = distance_membership_low
    # motion_detection.input['distance']['medium'] = distance_membership_medium
    # motion_detection.input['distance']['high'] = distance_membership_high

    # motion_detection.input['difference']['low'] = difference_membership_low
    # motion_detection.input['difference']['medium'] = difference_membership_medium
    # motion_detection.input['difference']['high'] = difference_membership_high

    # motion_detection.input['size']['little'] = size_membership_little
    # motion_detection.input['size']['medium'] = size_membership_medium
    # motion_detection.input['size']['big'] = size_membership_big
    
    ###############################################################################################
    
    # motion_detection.compute()
    # attention_result = motion_detection.output['attention_level']
    # print('attention level:', attention_result)
    
    ###############################################################################################
    
    # defuzzified_result = fuzz.defuzz(attention_level.universe, attention_result, 'centroid')

    # print(f"Attention Level: {defuzzified_result}")
    
    # Check for termination
    if cv2.waitKey(40) == 27 or not ret :
        break
    
    old_background = np.multiply(new_background, 0.999)
    # result_image1 = old_background.astype(np.uint8)
    
    new_frame = np.multiply(gray_frame, 0.001)
    # result_image2 = new_frame.astype(np.uint8)
    
    new_background = old_background + new_frame
    
    new_b = new_background.astype(np.uint8)
    cv2.imshow("new_bacground", new_b)
    
alarm()
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()