import dlib
import time
import imutils
import math
import os
import numpy as np
from keras import backend as K
from keras.models import Model, load_model
import cv2
import json

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('one2.mp4')
overspeed=[]
underspeed=[]
data = []
carWidth = 1.3
SPEED = 15
WIDTH = 1280
HEIGHT = 950
fps = int(video.get(cv2.CAP_PROP_FPS))
print(fps)
letters= [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' ,'A', 'B', 'C' ,'H', 'J', 'K ','Y']

def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    print(location2[2])
    ppm = location2[2] / carWidth
    print(ppm)
    #ppm = 65
    d_meters = d_pixels / ppm
    print("d_pixels=" + str(d_pixels), "d_meters=" + str(d_meters))
    fps = 30
    speed = d_meters * fps * 3.6
    return speed


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000
    # Write output to video file
    out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))


    while True:
        start_time = time.time()

        rc, image = video.read()
        if type(image) == type(None):
            break

        #image = cv2.resize(image, (WIDTH, HEIGHT))
        image = image[1:1280, 150:950]
        #image = image[150:950, 150:1280]

        resultImage = image.copy()

        frameCounter = frameCounter + 1

        cv2.line(resultImage, (0, 300), (1280, 300), (255, 0, 0), 2)

        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(image)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            print('Removing carID ' + str(carID) + ' from list of trackers.')
            print('Removing carID ' + str(carID) + ' previous location.')
            print('Removing carID ' + str(carID) + ' current location.')
            carTracker.pop(carID, None)
            carLocation1.pop(carID, None)
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.11, 20, 18, (24, 24))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + 0.5 * w
                y_bar = y + 0.5 * h


                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + 0.5 * t_w
                    t_y_bar = t_y + 0.5 * t_h


                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (
                            x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    print('Creating new tracker ' + str(currentCarID))

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                    tracker.update(resultImage)

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1


        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())
            if t_w > 150:
                roi = cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
            #else:
             #   roi = cv2.rectangle(resultImage, (t_x, t_y), (t_x + 115, t_y + t_h), rectangleColor, 4)
            # speed estimation
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()
        #print(end_time)

        if not (end_time == start_time):
            fps = 1.0 / (end_time - start_time)


        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 300:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])
                        cv2.putText(resultImage, str(round(speed[i], 0)) + " km/hr", (int(x1 + w1 / 2), int(y1 - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        print('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
                        if speed[i] > SPEED:
                            cv2.imwrite('{}.png'.format(i), roi)
                            #cropped_image = image[y1:y1 + h1, x1 + 5:x1 + w1]

                            # overspeed.append(i)
                            # overspeed.append(speed[i])
                            #cv2.imwrite('Vehicles\{}.png'.format(i), cropped_image)
                            #cv2.imshow('cropped', cropped_image)
                        else:
                            cv2.imwrite('{}.png'.format(i), roi)
                            #cropped_image = image[y1:y1 + h1, x1 + 5:x1 + w1]

                            #underspeed.append(i)
                            #underspeed.append(speed[i])
                            #cv2.imwrite("Vehicles\{}.png".format(i), cropped_image)
                            #cv2.imshow('cropped', cropped_image)

        cv2.imshow('result', resultImage)
        # out.write(resultImage)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    trackMultipleObjects()