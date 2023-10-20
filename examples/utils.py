import cv2
import numpy as np


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


def crop(path):
    radius = 15
    color = (0, 0, 255)
    points = []

    image = cv2.imread(path + '/image.png')

    bbox = auto_bbox(image)
    points.append([bbox[0], bbox[1]])
    points.append([bbox[2], bbox[1]])
    points.append([bbox[0], bbox[3]])
    points.append([bbox[2], bbox[3]])

    dragging_point_index = None 

    def handle_mouse_events(event, x, y, flags, param):
        global dragging_point_index 

        if event == cv2.EVENT_LBUTTONDOWN:
            for i in range(len(points)):
                px, py = points[i]
                if abs(x - px) < radius and abs(y - py) < radius: 
                    dragging_point_index = i 

        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging_point_index == 0:
                points[dragging_point_index] = [x, y]
                points[2][0] = x
                points[1][1] = y
            elif dragging_point_index == 1:
                points[dragging_point_index] = [x, y]
                points[3][0] = x
                points[0][1] = y
            elif dragging_point_index == 2:
                points[dragging_point_index] = [x, y]
                points[0][0] = x
                points[3][1] = y
            elif dragging_point_index == 3:
                points[dragging_point_index] = [x, y]
                points[1][0] = x
                points[2][1] = y

        elif event == cv2.EVENT_LBUTTONUP:
            dragging_point_index= None

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setMouseCallback('image', handle_mouse_events)

    while True:
        image_copy = image.copy()
        cv2.rectangle(image_copy, (points[0][0], points[0][1], points[3][0] - points[0][0], points[3][1] - points[0][1]), color, 4)
        for point in points:
            cv2.circle(image_copy, point, radius, color, -1, cv2.LINE_AA)

        cv2.imshow('image', image_copy)

        if cv2.waitKey(1) & 0xFF == 27:
            break 

    cv2.destroyAllWindows()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image[points[0][1]:points[3][1], points[0][0]:points[3][0]]

    cv2.imwrite(path + '/texture.png', image)
    
    
def mask(path):
    
    oldx = oldy = 0
    color = [0, 255]
    thickness = 4
    
    src = cv2.imread(path + '/image.png')
    mask = cv2.imread(path + '/mask.png')
    mask_copy = mask.copy()
    
    def on_mouse(event, x, y, flags, param):

        global thickness, oldx, oldy

        if event == cv2.EVENT_LBUTTONDOWN:
            oldx, oldy = x, y

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                thickness += 1
            elif thickness > 1:
                thickness -= 1

        elif event == cv2.EVENT_MOUSEMOVE:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                cv2.line(mask_copy, (oldx, oldy), (x, y), (color[0], color[0], color[0]), thickness, cv2.LINE_AA)
                oldx, oldy = x, y
                
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.namedWindow('masked_image', cv2.WINDOW_NORMAL)
    
    cv2.setMouseCallback('mask', on_mouse)
    
    print()
    print("Press 'c' to change color")
    print("Press 'r' to reset mask")
    print("Turn the mouse wheel up to increase thickness")
    print("Turn the mouse wheel down to decrease thickness")
    print("Press 'esc' to finish")
    
    while True:
        dst = cv2.bitwise_and(src, mask_copy)
        cv2.imshow('mask', mask_copy)
        cv2.imshow('masked_image', dst)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == ord('c'):
            color = color[::-1]
            
        elif key == ord('r'):
            mask_copy = mask.copy()
            
    cv2.destroyAllWindows()
    cv2.imwrite(path + '/mask.png', mask_copy)