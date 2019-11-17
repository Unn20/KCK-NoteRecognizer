import numpy as np
import cv2
import math
import statistics
import scipy.ndimage as sc

class StaveRecogniter:
    def __init__(self, DEBUG=False):
        self.debugMode = DEBUG
        if self.debugMode:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 800, 800)

    def findRotAngle(self, lines):
        ''' Function that takes median from set of line's angles which are calculated on set of lines '''
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

        # Output is in degrees
        return statistics.median(angles)

    def checkForInverse(self, stave_thresholded):
        ''' Function that check amount of white pixels on both sides of picture and assume if stave must be inverted '''
        left = 0
        right = 0
        transposed = np.transpose(stave_thresholded)

        for i in range(math.floor(stave_thresholded.shape[1] / 4)):
                filledLeft = np.where(stave_thresholded[: , i] > 10.0)
                left += len(filledLeft[0])
                filledRight = np.where(stave_thresholded[: , stave_thresholded.shape[1] - (i+1)] > 10.0)
                right += len(filledRight[0])

        print(f"Left = {left} ; Right = {right}")
        # cv2.imshow('image', stave_thresholded)
        # cv2.waitKey()

        if 1.1 * left < right < 1.35 * left:
            return True
        else:
            return False


    def adjustGamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")

        return cv2.LUT(image, table)

    def groupLines(self, image):
        ''' Parse image from top to bottom, find lines and group them to staves '''
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
                        # print(f"Distances = {distance}")
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

    def run(self, image_input):
        ''' Main function that take an image input and crop out staves '''

        # Read an image
        img = cv2.imread(image_input)

        # Make a matrix copy with image shapes filled with zeroes
        matZero = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

        # Filter image with gauss blur
        gauss = cv2.GaussianBlur(img, (13,13), cv2.BORDER_DEFAULT)

        # Turn image into black and white colors
        gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)

        # Adjust gamma on image to extract darker and brighter elements
        gamma_adjusted = self.adjustGamma(gray, 1.2)

        # Adjust threshold on image to highly extract dark and bright elements
        thresholded = cv2.adaptiveThreshold(gamma_adjusted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 5)

        # Dilation and erosion to remove 'noises' on a picture
        img_dilation = cv2.dilate(thresholded, np.ones((5,5), np.uint8) , iterations=1) 
        img_erosion = cv2.erode(img_dilation, np.ones((5,5), np.uint8) , iterations=1) 

        # This parameter is used in Hough Transform function's and it depend's on image resolution
        minLine = (img.shape[0] + img.shape[1]) / 4

        # Do hough transform to initially extract lines
        lines = cv2.HoughLinesP(img_erosion, 1, np.pi/360, int(minLine / 2), minLineLength=minLine , maxLineGap=minLine/2)

        # Draw lines on black copy of image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(matZero, (x1, y1), (x2, y2), (255), 2)

        # Find angle to rotate an image and it's copy
        rotateAngle = self.findRotAngle(lines)

        thresholded_rotated = sc.rotate(img_erosion, rotateAngle)
        rotated = sc.rotate(matZero, rotateAngle)
        image_rotated = sc.rotate(img, rotateAngle)

        # Find staves from black image
        linesVertices = self.groupLines(rotated)

        if self.debugMode:
            cv2.imshow('image', img)
            cv2.waitKey()
            cv2.imshow('image', gauss)
            cv2.waitKey()
            cv2.imshow('image', gray)
            cv2.waitKey()
            cv2.imshow('image', gamma_adjusted)
            cv2.waitKey()
            cv2.imshow('image', thresholded)
            cv2.waitKey()
            cv2.imshow('image', img_erosion)
            cv2.waitKey()
            cv2.imshow('image', image_rotated)
            cv2.waitKey()
            cv2.imshow('image', rotated)
            cv2.waitKey()
        
        outputNo = 1
        for vertices in linesVertices:
            x1, y1 = vertices[0]
            x2, y2 = vertices[1]
            x3, y3 = vertices[2]
            x4, y4 = vertices[3]
            dist = vertices[4][0] * 3
            # print(f"x1 = {x1}, x3 = {x3}, y1 = {y1}, y3 = {y3}, distance = {dist}")
            # Find borders of cutted image
            # Left border
            if x1 - dist < 0:
                left = 0
            else:
                left = x1 - dist
            # Right border
            if x3 + dist > image_rotated.shape[1]:
                right = image_rotated.shape[1]
            else:
                right = x3 + dist
            # Up border
            if y1 - dist < 0:
                up = 0
            else:
                up = y1 - dist
            # Down border
            if y3 + dist > image_rotated.shape[0]:
                down = image_rotated.shape[0]
            else:
                down = y3 + dist
            #print(f"Result: left = {left}, right = {right}, up = {up}, down = {down}")
            #Cut an image from original
            staveThresholdedArray = thresholded_rotated[up:down, left:right]
            staveArray = image_rotated[up:down, left:right]

            if self.checkForInverse(staveThresholdedArray):
                outputArray = cv2.flip(staveArray, -1)
            else:
                outputArray = staveArray

            if self.debugMode:
                cv2.imshow('image', outputArray)
                cv2.waitKey()
            else:
                # Save results
                fileNameFragments = image_input.rsplit('.', 1)
                cv2.imwrite(fileNameFragments[0] + '_out' + str(outputNo) + '.' + fileNameFragments[1], outputArray)
            outputNo += 1

        print(f"Found {outputNo - 1} staves total.")

        print("Finish!")


if __name__ == '__main__':
    
    # Turn DEBUG to False if you want to save output, otherwise keep True
    sR = StaveRecogniter(DEBUG=True)

    sR.run("input/20.jpg")
    