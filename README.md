# Weapon-Detection-System
Weapon detection system with auto-alert functionality based on the Yolov7 algorithm, capable of identifying and flagging potential threats in real time



WEAPON DETECTION SYSTEM WITH AUTO ALERT SYSTEM

I have submitted four files  in which you need to run those file based on the following steps.
File 1 : Weapon_detection_system_with_auto_alert_system.ipynb
File 2 : Supporting file
File 3 : Mobile app

DETAILED EXPLANATION TO RUN OUR PROJECT

EXECUTION 1 : Weapon_detection_system_with_auto_alert_system.ipynb (file 1)
Environment : Google Colab 

Step 1 : First place our "weapon_dataset.zip"  in your google drive

Step 2 : Once the code has been run, you will receive the dataset and yolov7. You may find the newly generated yolov7 folder in the files area. Weapon_detection_dataset folder contains a dataset.yaml file, and you must modify the file location of the train and test by copying the location path of the train and test of the image folder. 

Step 3: Now run the remaining code exactly as is. 




EXECUTION 2 : VS code in Supporting file (file 2)
Environment : VS Code

Note : I have developed the code based on my system versions so it is best if your system has these versions of libraries to prevent any errors

Versions:
Python : 3.7.8
Torch : 1.8.0 
Torch vision : 0.9.0
Cv2 : 4.1.1.26

Step 1: To run our project open new terminal and enter the following command 
"python detect.py" 

Step 2: Download the mobile app in your android phone through " Weapon.apk " file in file 3

Step 3: After running your app login into the application

Login module:
Username : 1
Password : 1
IP address: "**This can be found while running the model***"
Notification Topic : WEAPON-NT
Shop topic : WEAPON-WT

Step 4 : Show objects which is in our classifier and our model predict it




