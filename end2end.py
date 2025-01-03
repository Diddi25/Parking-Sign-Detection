import easyocr
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

parking_information = []

#view test image

#pathToTestPicture = r"D:\OneDrive\Documents\Diddi\AI-projekt\yolov7\AI-Main-Project-3\test\images\p-skiva-skylt_jpg.rf.9cf6eba217156be500efb441efd91d90.jpg"

# Build absolute path dynamically
current_dir = os.getcwd()  # Gets the directory where the script is running
pathToTestPicture = os.path.join(current_dir, "yolov7", "AI-Main-Project-3", "test", "images", "p-skiva-skylt_jpg.rf.9cf6eba217156be500efb441efd91d90.jpg")

testImageWithoutRGB = cv2.imread(pathToTestPicture)
testImage = cv2.cvtColor(testImageWithoutRGB, cv2.COLOR_BGR2RGB)
plt.imshow(testImage)

#import reader

ocr = easyocr.Reader(['en'])
detectedTexts = ocr.readtext(testImage)

#print the detectedTexts

for text in detectedTexts:
    if("p" or "P" in text[1]):
        print(f"Bounding boxes: {text[0]}")
    print(f"Detected text: {text[1]}")  # Use f-string for better readability
    print(f"Confidence score: {text[2]:.2f}")  # Format confidence score to 2 decimal places

#collect data from detected_texts[]

bool_weekday_info = False
bool_saturday_info = False
bool_sunday_info = False

weekday_info = ""
saturday_info = ""
sunday_info = ""

timetable_info = ""
ovrig_info = ["\nÖvrig information:\n"]
disc_recuirement = False
disc_info ="You have to have a parking disc."

p_count = 0

#klassifiera information
for text in detectedTexts:
    if(len(text[1]) == 4):
        bool_weekday_info = True
        weekday_info = text[1]
    if(len(text[1]) > 4):
        bool_saturday_info = True
        saturday_info = text[1]
    if('tim' in text[1]):
        timetable_info = text[1]
    if('P' in text[1]):
        p_count+=1
        if(p_count == 2):
            disc_recuirement = True

print("\n")

now = datetime.now() #get current date:
current_date = now.strftime('%Y-%m-%d')
current_day = now.strftime('%A')
current_time = now.strftime('%H:%M')

print("\n")

parking_information.append("Based on the current date today: " + 
                           current_date + ", " +
                           current_day + ", " +
                           "at time: " + current_time + ":\n")

if(current_day == "Sunday"):
    if(bool_sunday_info):
        parking_information.append("You are allowed to park between: " + sunday_info + "\n")
elif(current_day == "Saturday"):
    if(bool_saturday_info):
        parking_information.append("You are allowed to park between: " + saturday_info + "\n")
else:
    if(bool_weekday_info):
        parking_information.append("You are allowed to park between: " + weekday_info + ".\n")

if timetable_info:
    parking_information.append("You can park up to: " + timetable_info + "mar.\n")
    if(weekday_info or saturday_info or sunday_info):
        ovrig_info.append("Utanför tidsramen " + weekday_info + " kan du parkera längre än " + timetable_info + "mar.\n" )

if disc_recuirement:
    parking_information.append(disc_info + "\n")
    if(weekday_info or saturday_info or sunday_info):
        ovrig_info.append("Utanför tidsramen " + weekday_info + " behöver du inte ha parkeringsskiva. \n")

parking_information.extend(ovrig_info)

print("".join(parking_information))  #tar bort oönskade tecken

