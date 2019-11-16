import numpy as np
import cv2
import math
import statistics
import scipy.ndimage as sc

class Found(Exception) : pass

class StaveRecogniter:
    def __init__(self):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 800, 800)

    def findRotAngle(self, lines):
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x1 < x2:
                xLeft = x1
                yLeft = y1
                xRight = x2
                yRight = y2
            else:
                xLeft = x2
                yLeft = y2
                xRight = x1
                yRight = y1

            dy = yRight - yLeft
            dx = xRight - xLeft

            if dx != 0:
                angle = math.atan(dy/dx) # in pi
            else:
                angle = math.pi / 4
            angle = math.degrees(angle)
            angles.append(angle)

        return statistics.median(angles)

    def adjustGamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    def groupLines(self, image):
        output = []

        thickness = 1
        found = False
        yNo = 0
        line = 0
        xLeft = []
        xRight = []
        distance = []
        dist = 0

        for y in image:
            if np.quantile(y, 0.95) > 10.0:
                if not found:
                    found = True
                    line+= 1
                    
                    if line % 5 == 1:
                        yUp = yNo
                        distance.clear()
                        dist = 0
                    else:
                        distance.append(dist)
                        dist = 0

                    if line % 5 == 0:
                        yDown = yNo
                        # print(f"yUp = {yUp}")
                        # print(f"yDown = {yDown + thickness}")
                        # print(f"xLeft = {np.amin(xLeft)}")
                        # print(f"xRight = {np.amax(xRight)}")
                        # print(f"Thickness = {thickness}")
                        # print(f"Distance = {np.median(distance)}")
                        good = True
                        for d in distance:
                            if d > np.median(distance) * 5:
                                good = False

                        if good:
                            output.append([[int(np.amin(xLeft)), int(yUp)], [int(np.amax(xRight)), int(yUp)],
                                        [int(np.amax(xRight)), int(yDown)], [int(np.amin(xLeft)), int(yDown)], [int(np.median(distance))]])
                        xLeft.clear()
                        xRight.clear()

                    thickness = 1
                else:
                    thickness += 1
                    filled = np.where(y > 10.0)
                    xLeft.append(filled[0][0])
                    xRight.append(filled[0][-1])

            else:
                found = False
                dist += 1

            yNo+=1

        return output

    def houghTransform(self):
        img = cv2.imread("5.jpg")

        matZero = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        matZero2 = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)


        gauss = cv2.GaussianBlur(img, (13,13), cv2.BORDER_DEFAULT)

        gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)

        gamma_adjusted = self.adjustGamma(gray, 1.2)

        thresholded = cv2.adaptiveThreshold(gamma_adjusted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 5)


        img_dilation = cv2.dilate(thresholded, np.ones((5,5), np.uint8) , iterations=1) 
        img_erosion = cv2.erode(img_dilation, np.ones((5,5), np.uint8) , iterations=1) 

        minLine = (img.shape[0] + img.shape[1]) / 4

        print(minLine)


        lines = cv2.HoughLinesP(img_erosion, 1, np.pi/360, int(minLine / 2), minLineLength=minLine , maxLineGap=minLine/2)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(matZero, (x1, y1), (x2, y2), (255), 2)

        #edges = cv2.Canny(matZero, 20, 230, apertureSize=3)

        # print(f"angle = {self.findRotAngle(lines)}")

        rotateAngle = self.findRotAngle(lines)

        rotated = sc.rotate(matZero, rotateAngle)

        image_rotated = sc.rotate(img, rotateAngle)


        linesVertices = self.groupLines(rotated)
        

        for vertices in linesVertices:
            x1, y1 = vertices[0]
            x2, y2 = vertices[1]
            x3, y3 = vertices[2]
            x4, y4 = vertices[3]
            dist = vertices[4][0] * 3
            # print(f"x1 = {x1}, x3 = {x3}, y1 = {y1}, y3 = {y3}, distance = {dist}")
            # Left border
            if x1 - dist < 0:
                left = 0
            else:
                left = x1 - dist
            # Right border
            if x3 + dist > img.shape[1]:
                right = img.shape[1]
            else:
                right = x3 + dist
            # Up border
            if y1 - dist < 0:
                up = 0
            else:
                up = y1 - dist
            # Down border
            if y3 + dist > img.shape[0]:
                down = img.shape[0]
            else:
                down = y3 + dist
            #print(f"Result: left = {left}, right = {right}, up = {up}, down = {down}")
            lineArray = image_rotated[up:down, left:right]
            cv2.imshow('image', lineArray)
            cv2.waitKey()
            
        cv2.imshow('image', rotated)
        cv2.waitKey()
        

        # cv2.imshow('image', matZero)
        # cv2.waitKey()
        print("end")

if __name__ == '__main__':
    
    sR = StaveRecogniter()

    sR.houghTransform()
    
