import cv2 as cs2
import os as os
from os import listdir
from os.path import isfile, join
import json as json


originalPathData = "./dataset/train"
originalPathMarkup = "./markup/train"

outputPathData = "./set1/data/train"
outputPathMarkup = "./set1/markup/train"

MAX_HEIGHT = 42
MAX_WIDTH = 70

proeb = 0



for index in range(1, 68):
    filesData = [f for f in listdir(originalPathData + str(index))]
    filesMarkup = [f for f in listdir(originalPathMarkup + str(index))]
    
    print(len(filesData), len(filesMarkup))
    
    filesData.sort()
    filesMarkup.sort()

    maxE = len(filesData)

    outputPathDataName = outputPathData + str(index)
    outputPathMarkupName = outputPathMarkup + str(index)
    try:
        os.mkdir(outputPathDataName)
        os.mkdir(outputPathMarkupName)
    except:
        pass

    for e in range(0, maxE):
        fileName, markupName = filesData[e], filesMarkup[e]
        openDataName = originalPathData + str(index) + "/" + fileName
        openMarkupName = originalPathMarkup + str(index) + "/" + markupName
        img = cv2.imread(openDataName)
        with open(openMarkupName) as f:
            markup = json.load(f)
        height, width = img.shape[:2]

        scalingFactor = MAX_HEIGHT / float(height)
        img = cv2.resize(img, None, fx=scalingFactor, fy=scalingFactor, interpolation=cv2.INTER_AREA)

        _, newWidth = img.shape[:2]
        padding = MAX_WIDTH - newWidth
        if padding > 0:
            if padding % 2 == 0:
                image = cv2.copyMakeBorder(img, 0, 0, padding // 2, padding // 2, cv2.BORDER_CONSTANT)
            else:
                image = cv2.copyMakeBorder(img, 0, 0, padding // 2, (padding // 2) + 1, cv2.BORDER_CONSTANT)

            coordinatesArray = markup["plates"]
            for struct in coordinatesArray:
                for coords in struct["frame"]:
                    coords[0] = coords[0] * scalingFactor + padding // 2
                    coords[1] = coords[1] * scalingFactor


            cv2.imwrite(outputPathDataName + "/" + "car" + str(e) + '.png', image)
            with open(outputPathMarkupName + "/" + "carMarkup" + str(e) + ".json", 'w') as f:
                json.dump(markup, f)

            print("done", e)
        else:
            proeb += 1
            print("proeb")


print("total Proeb", proeb)

#cv2.destroyAllWindows()




def resize(img):
    height, width = img.shape[:2]
        
    scalingFactor = MAX_HEIGHT / float(height)
    img = cv2.resize(img, None, fx=scalingFactor, fy=scalingFactor, interpolation=cv2.INTER_AREA)
        
    _, newWidth = img.shape[:2]
    padding = MAX_WIDTH - newWidth
    if padding > 0:
        if padding % 2 == 0:
            image = cv2.copyMakeBorder(img, 0, 0, padding // 2, padding // 2, cv2.BORDER_CONSTANT)
        else:
            image = cv2.copyMakeBorder(img, 0, 0, padding // 2, (padding // 2) + 1, cv2.BORDER_CONSTANT)
    return img
