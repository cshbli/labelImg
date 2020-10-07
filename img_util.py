import cv2
import numpy as np

def change_brightness(img, scale):
    new_img = np.where(img * scale < 255, img * scale, 255)
    return new_img.astype(img.dtype)

# given image and a list of boxes, create a new image that contains only the patches defined by the boxes, while
# the background is set to a given value
def patch_back(img, boxes, background=0):
    patches = []
    for xmin, ymin, xmax, ymax in boxes:
        patches.append((xmin, ymin, xmax, ymax, img[xmin:(xmax+1), ymin:(ymax+1)]))
    new_img = np.full(img.shape, background, dtype=img.dtype)
    for xmin, ymin, xmax, ymax, patch in patches:
        new_img[xmin:(xmax+1), ymin:(ymax+1)] = patch
    return new_img       
    
def test():
    img = cv2.imread('c:\Projects\Datasets\PLT\data\Images\IMG_3182_1_1_2_2.png')
    img1 = change_brightness(img, 1.4)
    img2 = change_brightness(img, 0.4)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.imshow('img x 1.4', img1)
    cv2.waitKey(0)
    cv2.imshow('img x 0.4', img2)
    cv2.waitKey(0)
    img3 = patch_back(img, [(0, 0, 128, 128), (129, 129, 511, 511)], 0)
    cv2.imshow('img3', img3)
    cv2.waitKey(0)
    
    cv2.destroyAllWindows()

test()