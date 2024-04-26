
Object Detection and Attention Level Recognition with Fuzzy Logic

This Python script implements a real-time object detection and attention level recognition system using fuzzy logic. The system is designed to monitor a video stream and identify objects based on their size, distance from the camera, and the difference in gray level compared to the background. Fuzzy logic is then used to determine the attention level based on these factors.

Features:

Real-time object detection Fuzzy logic for attention level recognition Uses background subtraction for object detection Generates a binary image to highlight objects Plays an alarm sound when attention level is high 

Requirements:

OpenCV (cv2) NumPy (np) skfuzzy playsound (optional) send_email (optional) 

How to Use:

Make sure you have the required libraries installed (cv2, numpy, skfuzzy). Place the script and any necessary audio files (e.g., alarm sound) in the same directory. Run the script using Python (python your_script_name.py). The script will display the original video stream, the binary image highlighting objects, and the calculated attention level. An alarm sound will play (if playsound is installed) when the attention level is high. 

Notes:

The script currently uses a pre-defined background image. You can modify the code to capture the background automatically during the first few seconds of the video. The fuzzy membership functions and rules can be further customized to better suit your specific application. The send_email function is currently not implemented but can be added to send email notifications when the attention level is high. 

Further Development:

Implement background capture during runtime. Refine the fuzzy membership functions and rules for better accuracy. Integrate email notification using the send_email function. Explore object classification techniques to identify specific object types. 

I hope this READ me helps!

