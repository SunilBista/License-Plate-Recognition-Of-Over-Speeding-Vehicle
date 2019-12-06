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
                    if (speed[i] == None or speed[i] == 0) and y1 >= 330:
                        speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2])
                        cv2.putText(resultImage, str(round(speed[i], 0)) + " km/hr", (int(x1 + w1 / 2), int(y1 - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        print('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
                        if speed[i] > SPEED:
                            cv2.imwrite('{}.png'.format(i), roi)
                            #cropped_image = image[y1:y1 + h1 + 25, x1:x1 + w1 + 25]
                            cropped_image = image[y1 + 200:y1 + h1 - 50, x1:x1 + w1 - 50]
                            overspeed.append(i)
                            overspeed.append(speed[i])
                            cv2.imwrite('Vehicles\{}.png'.format(i), cropped_image)
                            #cv2.imshow('cropped', cropped_image)
                        else:
                            cv2.imwrite('{}.png'.format(i), roi)
                            #cropped_image = image[y1:y1 + h1 + 25, x1:x1 + w1 + 25]
                            cropped_image = image[y1 + 200:y1 + h1 - 50, x1:x1 + w1 - 50]
                            underspeed.append(i)
                            underspeed.append(speed[i])
                            cv2.imwrite("Vehicles\{}.png".format(i), cropped_image)
                            #cv2.imshow('cropped', cropped_image)
                            #localization

                        cars_path = r'C:\Users\USER\Downloads\Compressed\final project\final project\Vehicles'
                        for filename in os.listdir(cars_path):
                            filename, _ = os.path.splitext(filename)
                            filepath = filename + ".png"
                            img = cv2.imread(filepath, cv2.IMREAD_COLOR)
                            img = cv2.resize(img, (1080, 720))
                            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grey scale

                            # bilateralFilter(InputArray src, OutputArray dst, int d, double sigmaColor, double sigmaSpace, int borderType=BORDER_DEFAULT )

                            gray = cv2.bilateralFilter(gray, 9, 17, 17)  # Blur to reduce noise

                            v = np.median(gray)
                            sigma = 0.33  # sigma=0.33  tends to give good results on most of the dataset

                            lower_thresh = int(max(0, (1.0 - sigma) * v))
                            upper_thresh = int(min(255, (1.0 + sigma) * v))

                            edged = cv2.Canny(gray, lower_thresh, upper_thresh)  # Perform Edge detection

                            # cv2.imshow('canny edged', edged)
                            #print('canny edge', edged)

                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                            # find contours in the edged image
                            # keep only the largest ones, and initialize our screen contour
                            cont = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            # CV_RETR_TREE retrieves all of the contours and reconstructs a full hierarchy of nested contours
                            # Compress horizontal, vertical, and diagonal segments, that is, the function leaves only their ending points

                            cont = imutils.grab_contours(cont)
                            # with reference to different version of opencv

                            cont = sorted(cont, key=cv2.contourArea, reverse=True)[:10]
                            # Only the 10 largest contours.
                            # key-> A Function to execute to decide the order.
                            # reverse-> A Boolean. False will sort ascending, True will sort descending.

                            screenCnt = None
                            # contour that corresponds to the number plate
                            # loop over our contours
                            for c in cont:
                                # approximate the contour
                                peri = cv2.arcLength(c, True)
                                epsilon = 0.018 * peri
                                approx = cv2.approxPolyDP(c, 0.018 * peri, True)

                                # cv2.drawContours(img, cont, -1, (0, 255, 0), 2)
                                # cv2.drawContours(img, approx, -1, (0, 0, 255), 2)

                                # for approximated contour to have four points, then
                                # we can assume that we have found our screen
                                if len(approx) == 4:
                                    screenCnt = approx
                                    break

                            if screenCnt is None:
                                detected = 0
                                print("No contour detected")
                            else:
                                detected = 1

                            if detected == 1:
                                cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
                            continue

                                # cv2.imshow('Contour', img)
                                # cv2.waitKey(0)
                                # cv2.destroyAllWindows()

                            # Masking the part other than the number plate
                            mask = np.zeros(gray.shape, np.uint8)  # uint8-> Unsigned integer (0 to 255)
                            # cv2.imshow('result', mask)

                            mask_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, 1)
                            # cv2.imshow('image', mask_image)

                            mask_image = cv2.bitwise_and(img, img, mask=mask)
                            # cv2.imshow('image', mask_image)

                            # a minEnclosingCircle in blue
                            for c in cont:
                                # get the bounding rect
                                x, y, w, h = cv2.boundingRect(c)
                                # draw a green rectangle to visualize the bounding rect
                                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                                # get the min area rect
                                rect = cv2.minAreaRect(c)
                                box = cv2.boxPoints(rect)
                                # convert all coordinates floating point values to int
                                box = np.int0(box)
                                # draw a red 'nghien' rectangle
                                cv2.drawContours(img, [box], 0, (0, 0, 255))

                                # finally, get the min enclosing circle
                                (x, y), radius = cv2.minEnclosingCircle(c)
                                # convert all values to int
                                center = (int(x), int(y))
                                radius = int(radius)
                                # and draw the circle in blue
                                img = cv2.circle(img, center, radius, (255, 0, 0), 2)

                            #print(len(cont))
                            cv2.drawContours(img, cont, -1, (255, 255, 0), 1)
                            # cv2.imshow("Encircled", img)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()

                            # Cropping
                            (x, y) = np.where(mask == 255)
                            (topx, topy) = (np.min(x), np.min(y))
                            (bottomx, bottomy) = (np.max(x), np.max(y))
                            Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]
                            Cropped = cv2.resize(Cropped, (400, 200))
                            cv2.imwrite('licenseplate\{}.png'.format(i), Cropped)
                            #cv2.imshow('Cropped', Cropped)
        # Write the frame into the file 'output.avi'
        cv2.imshow('result', resultImage)
        #out.write(resultImage)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    trackMultipleObjects()

model = load_model('FINAL_MODEL.h5', compile=False)

input_data = model.get_layer('the_input').output
y_pred = model.get_layer('softmax').output
model_p = Model(inputs=input_data, outputs=y_pred)

#!pip install scipy==1.1.0

from scipy.misc import imread, imresize
#use width and height from your neural network here.

def load_for_nn(img_file):
    image = imread(img_file, flatten=True)
    image = imresize(image,(64, 128))
    image = image.T

    images = np.ones((1,128,64)) #change 1 to any number of images you want to predict, here I just want to predict one
    images[0] = image
    images = images[:,:,:,np.newaxis]
    images /= 255

    return images

def predict_image(image_path): #insert the path of your image
    image = load_for_nn(image_path) #load from the snippet code
    raw_word = model_p.predict(image) #do the prediction with the neural network
    #final_word = decode_output(raw_word)[0] #the output of our neural network is only numbers. Use decode_output from image_ocr.py to get the desirable string.
    return raw_word
def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))

def decode_predict_ctc(out, top_paths = 1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
      beam_width = top_paths
    for i in range(top_paths):
      lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0])*out.shape[1],
                           greedy= True, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
      text = labels_to_text(lables)
      results.append(text)
    return results


cars_path = r'C:\Users\USER\Downloads\Compressed\final project\final project\Vehicles'
for filename in os.listdir(cars_path):
    filename, _ = os.path.splitext(filename)
    filepath = filename + ".png"
    x = predict_image(filepath)
    pred_texts = decode_predict_ctc(x)
    print(pred_texts)
    if int(filename) in underspeed:
        j = underspeed.index(int(filename))
        data.append({'Car_ID': filename, 'Speed': underspeed[j+1], 'Description':'Underspeed', "License plate number": pred_texts})
        filepath = filename + ".json"
        with open(filepath, 'w') as outfile:
            json.dump(data,outfile)
    else:
        j = overspeed.index(int(filename))
        data.append({'Car_ID': filename, 'Speed': overspeed[j + 1], 'Description': 'overspeed',
                     "License plate number": pred_texts})

filepath = 'data' + ".json"
with open(filepath, 'w') as outfile:
        json.dump(data, outfile)














