from mss import mss
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pytesseract
import pynput

with mss() as sct:

    monitor = {'left': 0, 'top': 70, 'width': 1728, 'height': 1117}
    sct_img = sct.grab(monitor)
    shot = np.array(Image.frombytes(
        'RGB', sct_img.size, sct_img.bgra, 'raw', 'BGRX'))

    shot = cv2.cvtColor(shot, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(shot, cv2.COLOR_BGR2GRAY)
    self_life_count = shot[100:200, 3000:3100]
    opponent_life_count = shot[100:200, 3200:3280]
    self_health = shot[150, 3175]
    opponent_health = shot[160, 3345]
    cv2.imshow('window', opponent_life_count)

    print('Tesseract')
    print(pytesseract.image_to_string(opponent_life_count, lang='eng',
                                      config='--psm 10 --oem 3 -c tessedit_char_whitelist=-10123456789'))
    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
    print(self_health[2], sum(self_health))
    print(gray.shape)
