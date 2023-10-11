import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='black', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='black', linewidth=1.25)   
    
    
def show_box(box, ax, lw=2):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=lw))    

    
def show_mask(mask, ax):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1)
    ax.imshow(mask_image, cmap='gray')
    

def auto_bbox(image, th=160):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = image.shape[0]
    w = image.shape[1]
    point = []
    for _ in range(4):
        x = y = 0
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i,j,0] > th:
                    x, y = i, j
                    break
            if x != 0:
                break

        point.append([y, x])
        image = np.rot90(image)
    
    point[1] = [w - point[1][1], point[1][0]]
    point[2] = [w - point[2][0], h - point[2][1]]
    point[3] = [point[3][1], h - point[3][0]]
    
    x1, y1, x2, y2 = point[3][0], point[0][1], point[1][0], point[2][1]
    
    if x1 > 20:
        x1 -= 20
    
    if y1 > 20:
        y1 -= 20
    
    if w - x2 > 20:
        x2 += 20
    
    if h - y2 > 20:
        y2 += 20
    
    bbox = np.array([x1, y1, x2, y2])
    return bbox