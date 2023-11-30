import os
import cv2
import numpy as np
from sklearn import preprocessing
from skimage.feature import hog
from skimage import exposure
from progress.bar import Bar
import time

def main():
    mainStartTime = time.time()
    trainImagePath = './images_split/train/'
    testImagePath = './images_split/test/'
    trainFeaturePath = './features_labels/train/'
    testFeaturePath = './features_labels/test/'
    print(f'[INFO] ========= TRAINING IMAGES ========= ')
    trainImages, trainLabels = getData(trainImagePath)
    trainEncodedLabels, encoderClasses = encodeLabels(trainLabels)
    trainFeatures = extractHOGFeatures(trainImages)
    saveData(trainFeaturePath, trainEncodedLabels, trainFeatures, encoderClasses)
    print(f'[INFO] =========== TEST IMAGES =========== ')
    testImages, testLabels = getData(testImagePath)
    testEncodedLabels, encoderClasses = encodeLabels(testLabels)
    testFeatures = extractHOGFeatures(testImages)
    saveData(testFeaturePath, testEncodedLabels, testFeatures, encoderClasses)
    elapsedTime = round(time.time() - mainStartTime, 2)
    print(f'[INFO] Code execution time: {elapsedTime}s')

def getData(path):
    images = []
    labels = []
    if os.path.exists(path):
        for dirpath, dirnames, filenames in os.walk(path):
            if len(filenames) > 0:
                folder_name = os.path.basename(dirpath)
                bar = Bar(
                    f'[INFO] Getting images and labels from {folder_name}',
                    max=len(filenames),
                    suffix='%(index)d/%(max)d Duration:%(elapsed)ds'
                )
                for index, file in enumerate(filenames):
                    label = folder_name
                    labels.append(label)
                    full_path = os.path.join(dirpath, file)
                    image = cv2.imread(full_path)
                    images.append(image)
                    bar.next()
                bar.finish()
        return images, np.array(labels, dtype=object)

def extractHOGFeatures(images):
    bar = Bar('[INFO] Extrating HOG features...', max=len(images), suffix='%(index)d/%(max)d  Duration:%(elapsed)ds')
    featuresList = []
    for image in images:
        if np.ndim(image) > 2:  # > 2 = colorida
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        featuresList.append(fd)
        bar.next()
    bar.finish()
    return np.array(featuresList, dtype=object)

def encodeLabels(labels):
    startTime = time.time()
    print(f'[INFO] Encoding labels to numerical labels')
    encoder = preprocessing.LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Encoding done in {elapsedTime}s')
    return np.array(encoded_labels, dtype=object), encoder.classes_

def saveData(path, labels, features, encoderClasses):
    startTime = time.time()
    print(f'[INFO] Saving data')
    label_filename = f'{labels=}'.split('=')[0] + '.csv'
    feature_filename = f'{features=}'.split('=')[0] + '.csv'
    encoder_filename = f'{encoderClasses=}'.split('=')[0] + '.csv'
    np.savetxt(path + label_filename, labels, delimiter=',', fmt='%i')
    np.savetxt(path + feature_filename, features, delimiter=',')
    np.savetxt(path + encoder_filename, encoderClasses, delimiter=',', fmt='%s')
    elapsedTime = round(time.time() - startTime, 2)
    print(f'[INFO] Saving done in {elapsedTime}s')

if __name__ == "__main__":
    main()
