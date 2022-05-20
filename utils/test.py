from mss import mss
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
import pynput
# 558 864
with mss() as sct:

    monitor = {'left': 0, 'top': 100, 'bottom':70, 'width': 3456, 'height': 2234}
    sct_img = sct.grab(monitor)
    shot = np.array(Image.frombytes(
        'RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX'))

    shot = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
    gray = gray[350:2050, 750:]
    self_life_count = shot[50:125, 3100:3160]
    opponent_life_count = shot[50:125, 3270:3330]
    self_health = shot[110, 3168]
    opponent_health = shot[110, 3338]
    cv2.imshow('window', self_health)

    gray = cv2.resize(gray, (0,0), fx=0.25, fy=0.25)

    print('Tesseract')
    print(pytesseract.image_to_string(opponent_life_count, lang='eng',
                                      config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789'))
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
    print(self_health, opponent_health)
    print(self_health[2], sum(self_health))
    print(opponent_health[2], sum(opponent_health))
    
    print(gray.shape)
