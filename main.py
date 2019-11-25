import numpy as np
import cv2
import math
import statistics
import scipy.ndimage as sc
import copy

class StaveRecogniter:

    # Jednolity rozmiar okna dla trybu debugowania
    def __init__(self, DEBUG=False):
        self.debugMode = DEBUG
        if self.debugMode:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 800, 800)

    # Zwraca medianę z kąta obrotów wykrytych linii (w pięcioliniach), w stopniach
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
                angle = math.atan(dy/dx) # w radianach
            else:
                angle = math.pi / 4
            angle = math.degrees(angle) # w stopniach
            angles.append(angle)

        return statistics.median(angles)

    # Zwraca 'true', gdy obraz jest obrócony do góry nogami, 'false' gdy jest dobrze
    def checkForInverse(self, stave_thresholded):
        left = 0
        right = 0
        transposed = np.transpose(stave_thresholded)

        for i in range(math.floor(stave_thresholded.shape[1] / 4)):
                filledLeft = np.where(stave_thresholded[: , i] > 10.0)
                left += len(filledLeft[0])
                filledRight = np.where(stave_thresholded[: , stave_thresholded.shape[1] - (i+1)] > 10.0)
                right += len(filledRight[0])
        if 1.1 * left < right < 1.35 * left:
            return True
        else:
            return False

    # Korekcja gamma obrazu
    def adjustGamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    # Zwraca listę granic pięciolinii i odstępów między liniami w pięciolinii
    #       Skanuje obraz od góry do dołu, znajduje linie i grupuje je w pięciolinie
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

    # Sortuje znalezione nuty po x-ach
    def sortCircles(self, circles, dist):
        for i in range(circles.shape[0]-2):
            for j in range(circles.shape[0]-1, i, -1):
                if circles[j-1, 0] > circles[j, 0]:
                    circles[j], circles[j-1] = copy.copy(circles[j-1]), copy.copy(circles[j])
        if circles[1,0] - circles[0,0] < dist/2:
            circles = circles[2:]
        return circles

    # Zapisuje wycięte pięciolinie jako osobne obrazy
    def run(self, image_input):
        img = cv2.imread(image_input+'.jpg')
        matZero = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8) # Macierz zer o romiarach obrazu
        gauss = cv2.GaussianBlur(img, (13,13), cv2.BORDER_DEFAULT)       # Wygładzony, rozmyty
        gray = cv2.cvtColor(gauss, cv2.COLOR_BGR2GRAY)                   # Szary
        gamma_adjusted = self.adjustGamma(gray, 1.2)                     # Lepszy kontrast
        thresholded = cv2.adaptiveThreshold(gamma_adjusted, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 5)
        img_dilation = cv2.dilate(thresholded, np.ones((5,5), np.uint8) , iterations=1) 
        img_erosion = cv2.erode(img_dilation, np.ones((5,5), np.uint8) , iterations=1)   # Usunięte artefakty

        minLine = (img.shape[0] + img.shape[1]) / 4
        lines = cv2.HoughLinesP(img_erosion, 1, np.pi/360, int(minLine / 2), minLineLength=minLine , maxLineGap=minLine/2)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(matZero, (x1, y1), (x2, y2), (255), 2)

        rotateAngle = self.findRotAngle(lines)      # Kąt obrotu dla pięciolinii
        thresholded_rotated = sc.rotate(img_erosion, rotateAngle)    # Binarny, obrócony
        rotated = sc.rotate(matZero, rotateAngle)                    # Obraz zerowy, obrócony
        image_rotated = sc.rotate(img, rotateAngle)                  # Oryginalny, obrócony

        # Find staves from black image  ??
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
        
        distances = []
        flip = []
        borders = []
        outputNo = 1
        for vertices in linesVertices:
            x1, y1 = vertices[0]
            x2, y2 = vertices[1]
            x3, y3 = vertices[2]
            x4, y4 = vertices[3]
            dist = vertices[4][0] * 3
            distances.append(dist)
            
            # Granice obszaru do wycięcia
            if x1 - dist < 0:                           # Lewy brzeg
                left = 0
            else:
                left = x1 - dist
            if x3 + dist > image_rotated.shape[1]:      # Prawy brzeg
                right = image_rotated.shape[1]
            else:
                right = x3 + dist
            if y1 - dist < 0:                           # Górny brzeg
                up = 0
            else:
                up = y1 - dist
            if y3 + dist > image_rotated.shape[0]:      # Dolny brzeg
                down = image_rotated.shape[0]
            else:
                down = y3 + dist
            borders.append((left, right, up, down))

            # Wycięcie pięciolinii
            staveThresholdedArray = thresholded_rotated[up:down, left:right]  # Binarny
            staveArray = image_rotated[up:down, left:right]                   # Oryginalny

            flip.append(self.checkForInverse(staveThresholdedArray))
            ifFlip = 0
            for i in flip:
                ifFlip=ifFlip+1 if i else ifFlip
            if ifFlip/len(flip) >= 0.5:
                outputArray = cv2.flip(staveArray, -1)
            else:
                outputArray = staveArray

            if self.debugMode:                                  # Zapis obrazów/wyświetlenie
                cv2.imshow('image', outputArray)
                cv2.waitKey()
            else:
                cv2.imwrite(image_input + '_out' + str(outputNo) + '.jpg', outputArray)
            outputNo += 1
        ifFlip = 0
        for i in flip:
            ifFlip=ifFlip+1 if i else ifFlip
        if ifFlip/len(flip) >= 0.5:
            image_rotated = cv2.flip(image_rotated, -1)

        print(f"Found {outputNo - 1} staves total.")


        number_of_note = 1
        for number, dist in enumerate(distances, start=1):
            # Odczyt pojedynczej, obróconej pięciolinii
            image = cv2.imread(image_input+'_out'+str(number)+'.jpg')

            # Rozmycie - "wygładzenie" artefaktów na obrazie. Konwersja do szarości. Threshold.
            outputArray = cv2.GaussianBlur(image, (13,13), cv2.BORDER_DEFAULT)
            outputArray = cv2.cvtColor(outputArray, cv2.COLOR_BGR2GRAY)
            outputArray = cv2.adaptiveThreshold(outputArray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 13, 5)
            
            # Wykrywanie linii tworzących pięciolinię
            #       min_x1: najmniejszy początek wykrytej linii
            minLine = (image.shape[0] + image.shape[1]) / 4
            lines = cv2.HoughLinesP(outputArray, 1, np.pi/360, int(minLine / 2), minLineLength=minLine , maxLineGap=minLine/2)
            min_x1, min_y1, max_y1 = image.shape[1], image.shape[0], 0
            
            # Usuwanie linii tworzących pięciolinię z obrazu
            if lines is not None:
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if x1 < min_x1:
                            min_x1 = x1
                        if y1 < min_y1:
                            min_y1 = y1
                        if y1 > max_y1 and x1 - min_x1 < dist/2:
                            max_y1 = y1
                        cv2.line(outputArray, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # border: najmniejszy początek linii + spodziewane miejsce na klucz
            dist = (max_y1 - min_y1)//4
            border = min_x1+3*dist

            # Detekcja klucza wiolinowego/basowego
            treble_clef = True           # True - wiolinowy, False - basowy
            whites = 0
            key_place = outputArray[:min_y1, min_x1:border]
            for i in key_place:
                for j in i:
                    if j == 255:
                        whites = whites+1
            key_place = outputArray[max_y1:, min_x1:border]
            for i in key_place:
                for j in i:
                    if j == 255:
                        whites = whites+1
            total = (min_y1 + image.shape[0] - max_y1)*(border - min_x1)
            if whites/total > 0.02:
                print("Wiolinowy")
            else:
                print("Basowy")
                treble_clef = False

            outputArray = np.array([[*a, *b] for a, b in zip(np.zeros((image.shape[0], border)), outputArray[:, border:])], dtype='uint8')

            # Usuwanie artefaktów po liniach
            outputArray = cv2.morphologyEx(outputArray, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))

            # Właściwe wykrywanie nut (jako kół)
            circles = cv2.HoughCircles(outputArray, cv2.HOUGH_GRADIENT, 4, image.shape[0]//4, param1=100, param2=50, minRadius=dist//2,maxRadius=2*dist//3)
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                circles = self.sortCircles(circles, dist)

                # Określa wysokość i długość nuty
                while len(circles) > 0:
                    #jaka wysokość nuty
                    x = circles[0][0]
                    y = circles[0][1]
                    r = circles[0][2]
                    places = range(min_y1, max_y1+5, dist//2)
                    nearby = [abs(y-cell) for cell in places]
                    d = 10
                    attempt1 = nearby.index(min(nearby))
                    if attempt1 % 2 == 0:
                        left = (x-dist//2, min_y1 + attempt1*dist//2)
                        right = (x+dist//2, min_y1 + attempt1*dist//2)
                        left_fragment = outputArray[left[1]-d:left[1]+d, left[0]-d:left[0]+d]
                        right_fragment = outputArray[right[1]-d:right[1]+d, right[0]-d:right[0]+d]
                        counter1 = 0
                        for row in left_fragment:
                            for cell in row:
                                if cell == 255:
                                    counter1 = counter1+1
                        for row in right_fragment:
                            for cell in row:
                                if cell == 255:
                                    counter1 = counter1+1
                    else:
                        up = (x, min_y1 + attempt1*dist//2-dist//2)
                        down = (x, min_y1 + attempt1*dist//2+dist//2)
                        up_fragment = outputArray[up[1]-d:up[1]+d, up[0]-d:up[0]+d]
                        down_fragment = outputArray[down[1]-d:down[1]+d, down[0]-d:down[0]+d]
                        counter1 = 0
                        for row in up_fragment:
                            for cell in row:
                                if cell == 255:
                                    counter1 = counter1+1
                        for row in down_fragment:
                            for cell in row:
                                if cell == 255:
                                    counter1 = counter1+1
                    nearby[attempt1] = 3*dist
                    attempt2 = nearby.index(min(nearby))
                    if attempt2 % 2 == 0:
                        left = (x-dist//2, min_y1 + attempt2*dist//2)
                        right = (x+dist//2, min_y1 + attempt2*dist//2)
                        left_fragment = outputArray[left[1]-d:left[1]+d, left[0]-d:left[0]+d]
                        right_fragment = outputArray[right[1]-d:right[1]+d, right[0]-d:right[0]+d]
                        counter2 = 0
                        for row in left_fragment:
                            for cell in row:
                                if cell == 255:
                                    counter2 = counter2+1
                        for row in right_fragment:
                            for cell in row:
                                if cell == 255:
                                    counter2 = counter2+1
                    else:
                        up = (x, min_y1 + attempt2*dist//2-dist//2)
                        down = (x, min_y1 + attempt2*dist//2+dist//2)
                        up_fragment = outputArray[up[1]-d:up[1]+d, up[0]-d:up[0]+d]
                        down_fragment = outputArray[down[1]-d:down[1]+d, down[0]-d:down[0]+d]
                        counter2 = 0
                        for row in up_fragment:
                            for cell in row:
                                if cell == 255:
                                    counter2 = counter2+1
                        for row in down_fragment:
                            for cell in row:
                                if cell == 255:
                                    counter2 = counter2+1
                    treble = ['F', 'E', 'D', 'C', 'H', 'A', 'G', 'F', 'E']
                    bass = ['A', 'G', 'F', 'E', 'D', 'C', 'H', 'A', 'G']

                    if treble_clef:
                        if counter1 > counter2:
                            note_value = treble[attempt1]
                        else:
                            note_value = treble[attempt2]
                    else:
                        if counter1 > counter2:
                            note_value = bass[attempt1]
                        else:
                            note_value = bass[attempt2]

                    # jaka długość nuty
                    check = image[y-5:y+5, x-5:x+5]
                    counter = 0
                    filled = False
                    staff = False
                    for row in check:
                        for cell in row:
                            if cell[1] < 80 and cell[2] < 80:
                                counter = counter+1
                    if counter/100 > 0.7:
                        filled = True

                    check = image[y-3*dist if y > 3* dist else 0:y, x+r-8:x+r+8]
                    counter = 0
                    for row in check:
                        for cell in row:
                            if cell[1] < 80 and cell[2] < 80:
                                counter = counter+1
                    if counter/(18*dist) > 0.4:
                        staff = True

                    double_note_circle = False
                    if len(circles) > 1:
                        if circles[1,0] - x < 3*dist//2:
                            double_note_circle = True

                    if double_note_circle:
                        note_length = '8'
                    elif filled:
                        note_length = '4'
                    elif staff:
                        note_length = '2'
                    else:
                        note_length = '1'

                    cv2.circle(image_rotated, (borders[number-1][0]+x, borders[number-1][2]+y), r, (0, 255, 0), 2) # last argument - width of the circle
                    cv2.rectangle(image_rotated, (borders[number-1][0]+x - 1, borders[number-1][2]+y - 1), (borders[number-1][0]+x + 1, borders[number-1][2]+y + 1), (0, 128, 255), -1)
                    cv2.putText(image_rotated, str(number_of_note)+'. '+note_value+' '+note_length, (borders[number-1][0]+x-dist//3, borders[number-1][2]+y-dist), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5, cv2.LINE_AA)
                    if double_note_circle:
                        circles = circles[2:]
                    else:
                        circles = circles[1:]
                    number_of_note = number_of_note+1

        cv2.namedWindow('output', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('output', 800, 800)
        cv2.imshow("output", image_rotated)
        cv2.waitKey(0)

        print("Finish!")


if __name__ == '__main__':
    
    sR = StaveRecogniter(DEBUG=False)           # False = zapis obrazów, True = wyświetlenie obrazów
    sR.run("input/3")
    