import cv2

file_name = '/Users/alistair.curtis/Projects/CustodianFile/SampleVideos/6-6-24 john blackhawk real time.mp4'
cap = cv2.VideoCapture(file_name)

if not cap.isOpened():
    print("Error opening video file.")
else:
    print("Video file opened successfully.")
    cap.release()