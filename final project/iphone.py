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

letters = [' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'H', 'J', 'K', 'Y']

carCascade = cv2.CascadeClassifier('myhaar.xml')
video = cv2.VideoCapture('1.MOV')
overspeed = []
underspeed = []
data = []
carWidth = 1.3
SPEED = 10
WIDTH = 1280
HEIGHT = 950
fps = int(video.get(cv2.CAP_PROP_FPS))
print(fps)


def estimateSpeed(location1, location2):
    d_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    print(location2[2])
    ppm = location2[2] / carWidth
    print(ppm)
    # ppm = 65
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

        # image = cv2.resize(image, (WIDTH, HEIGHT))
        #image = image[1:1280, 150:950]
        image = image[90:1280, 600:1400]

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
                # else:
                #   roi = cv2.rectangle(resultImage, (t_x, t_y), (t_x + 115, t_y + t_h), rectangleColor, 4)
            # speed estimation
            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()
        # print(end_time)

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
                        # cv2.putText(resultImage, str(round(speed[i], 0)) + " km/hr", (int(x1 + w1 / 2), int(y1 - 5)),cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
                        print('CarID ' + str(i) + ': speed is ' + str("%.2f" % round(speed[i], 0)) + ' km/h.\n')
                        if speed[i] > SPEED:
                            cv2.imwrite('{}.png'.format(i), roi)
                            # cropped_image = image[y1:y1 + h1, x1 + 5:x1 + w1]

                            overspeed.append(i)
                            overspeed.append(speed[i])
                            cv2.imwrite('overspeed\{}.png'.format(i), roi)
                            # cv2.imshow('cropped', cropped_image)
                        else:
                            cv2.imwrite('{}.png'.format(i), roi)
                            # cropped_image = image[y1:y1 + h1, x1 + 5:x1 + w1]

                            underspeed.append(i)
                            underspeed.append(speed[i])
                            cv2.imwrite("normalspeed\{}.png".format(i), roi)
                            # cv2.imshow('cropped', cropped_image)

                        cars_path = r'C:\Users\USER\Downloads\Compressed\final project\final project\overspeed'
                        for filename in os.listdir(cars_path):
                            filename, _ = os.path.splitext(filename)
                            filepath = filename + ".png"

                            def detectPlateRough(image_gray, resize_h=720, en_scale=1.08, top_bottom_padding_rate=0.05):
                                if top_bottom_padding_rate > 0.2:
                                    print("error:top_bottom_padding_rate > 0.2:", top_bottom_padding_rate)
                                    exit(1)
                                height = image_gray.shape[0]
                                padding = int(height * top_bottom_padding_rate)
                                scale = image_gray.shape[1] / float(image_gray.shape[0])
                                image = cv2.resize(image_gray, (int(scale * resize_h), resize_h))
                                image_color_cropped = image[padding:resize_h - padding, 0:image_gray.shape[1]]
                                image_gray = cv2.cvtColor(image_color_cropped, cv2.COLOR_RGB2GRAY)
                                watches = watch_cascade.detectMultiScale(image_gray, en_scale, 2, minSize=(36, 9),
                                                                         maxSize=(36 * 40, 9 * 40))
                                cropped_images = []
                                for (x, y, w, h) in watches:
                                    # cv2.rectangle(image_color_cropped, (x, y), (x + w, y + h), (0, 0, 255), 1)

                                    x -= w * 0.14
                                    w += w * 0.28
                                    y -= h * 0.15
                                    h += h * 0.3

                                    # cv2.rectangle(image_color_cropped, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 1)

                                    cropped = cropImage(image_color_cropped, (int(x), int(y), int(w), int(h)))
                                    cropped_images.append([cropped, [x, y + padding, w, h]])

                                    # cv2.imshow("imageShow", cropped)
                                    # cv2.waitKey(0)
                                return cropped_images

                            def cropImage(image, rect):
                                cv2.imshow("imageShow", image)

                                x, y, w, h = computeSafeRegion(image.shape, rect)
                                im = image[y:y + h, x:x + w]
                                # cv2.imshow("imageShow", im)
                                cv2.imwrite('licenseplate\{}.png'.format(i), im)

                                return image[y:y + h, x:x + w]

                            def computeSafeRegion(shape, bounding_rect):
                                top = bounding_rect[1]  # y
                                bottom = bounding_rect[1] + bounding_rect[3]  # y +  h
                                left = bounding_rect[0]  # x
                                right = bounding_rect[0] + bounding_rect[2]  # x +  w
                                min_top = 0
                                max_bottom = shape[0]
                                min_left = 0
                                max_right = shape[1]

                                # print(left,top,right,bottom)
                                # print(max_bottom,max_right)

                                if top < min_top:
                                    top = min_top
                                if left < min_left:
                                    left = min_left
                                if bottom > max_bottom:
                                    bottom = max_bottom
                                if right > max_right:
                                    right = max_right
                                return [left, top, right - left, bottom - top]

                            watch_cascade = cv2.CascadeClassifier('cascade.xml')
                            oimage = cv2.imread(filepath)
                            limages = detectPlateRough(oimage, image.shape[0], top_bottom_padding_rate=0.1)

        cv2.imshow('result', resultImage)
        # out.write(resultImage)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    trackMultipleObjects()

model = load_model('FINAL_MODEL.h5', compile=False)

input_data = model.get_layer('the_input').output
y_pred = model.get_layer('softmax').output
model_p = Model(inputs=input_data, outputs=y_pred)

# !pip install scipy==1.1.0

from scipy.misc import imread, imresize


# use width and height from your neural network here.

def load_for_nn(img_file):
    image = imread(img_file, flatten=True)
    image = imresize(image, (64, 128))
    image = image.T

    images = np.ones(
        (1, 128, 64))  # change 1 to any number of images you want to predict, here I just want to predict one
    images[0] = image
    images = images[:, :, :, np.newaxis]
    images /= 255

    return images


def predict_image(image_path):  # insert the path of your image
    image = load_for_nn(image_path)  # load from the snippet code
    raw_word = model_p.predict(image)  # do the prediction with the neural network
    # final_word = decode_output(raw_word)[0] #the output of our neural network is only numbers. Use decode_output from image_ocr.py to get the desirable string.
    return raw_word


def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[int(x)], labels)))


def decode_predict_ctc(out, top_paths=1):
    results = []
    beam_width = 5
    if beam_width < top_paths:
        beam_width = top_paths
    for i in range(top_paths):
        lables = K.get_value(K.ctc_decode(out, input_length=np.ones(out.shape[0]) * out.shape[1],
                                          greedy=True, beam_width=beam_width, top_paths=top_paths)[0][i])[0]
        text = labels_to_text(lables)
        results.append(text)
    return results


cars_path = r'C:\Users\USER\Downloads\Compressed\final project\final project\licenseplate'
for filename in os.listdir(cars_path):
    filename, _ = os.path.splitext(filename)
    filepath = filename + ".png"
    x = predict_image(filepath)
    pred_texts = decode_predict_ctc(x)
    print(pred_texts)
    if int(filename) in underspeed:
        j = underspeed.index(int(filename))
        data.append({'Car_ID': filename, 'Speed': underspeed[j + 1], 'Description': 'Underspeed',
                     "License plate number": pred_texts})
        filepath = filename + ".json"
        with open(filepath, 'w') as outfile:
            json.dump(data, outfile)
    else:
        j = overspeed.index(int(filename))
        data.append({'Car_ID': filename, 'Speed': overspeed[j + 1], 'Description': 'overspeed',
                     "License plate number": pred_texts})

filepath = 'data' + ".json"
with open(filepath, 'w') as outfile:
    json.dump(data, outfile)
