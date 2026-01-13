import cv2
import numpy as np

# Variables globales para funciones de cropping
x_c, y_c = 0, 0

def mouse_crop(event, x, y, flags, param):
    """Mouse callback for cropping"""
    global x_c, y_c
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates of pixel: X: {x}, Y: {y}")
        x_c, y_c = x, y

def crop_single(image):
    """
    Single interactive crop
    
    Args:
        image: Input image
        
    Returns:
        tuple: (y, x) coordinates
    """
    alpha = 1  # Contrast control
    beta = 1   # Brightness control
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", mouse_crop)

    while True:
        key = cv2.waitKey(2)
        cv2.imshow("image", image)

        if key % 256 == 27:  # ESC key
            cv2.destroyAllWindows()
            break

    return y_c, x_c

def crop(image, ancho, largo, n):
    """Interactive cropping of multiple regions"""
    Is = []
    cord = []
    alpha = 1  # Contrast control
    beta = 1   # Brightness control
    croped = image
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    for i in range(n):
        cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("image", mouse_crop)

        while True:
            key = cv2.waitKey(2)
            cv2.imshow("image", image)
            
            if key % 256 == 27:
                cv2.destroyAllWindows()
                break

        Is.append(np.mean(cv2.fastNlMeansDenoising(
            image[y_c-2*ancho:y_c, x_c-largo:x_c+largo]), axis=2) * 1.0)
        Is.append(np.mean(croped[y_c-2*ancho:y_c, x_c-largo:x_c+largo], axis=2) * 1.0)
        cord.append([y_c, x_c])
        image[y_c-2*ancho:y_c, x_c-largo:x_c+largo] = 0

    return Is, cord

def ROI(image_name):
    """Select Region of Interest"""
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    cropping = False
    
    def mouse_crop_local(event, x, y, flags, param):
        nonlocal x_start, y_start, x_end, y_end, cropping
        
        if event == cv2.EVENT_LBUTTONDOWN:
            x_start, y_start, x_end, y_end = x, y, x, y
            cropping = True
        elif event == cv2.EVENT_MOUSEMOVE:
            if cropping:
                x_end, y_end = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            x_end, y_end = x, y
            cropping = False
            print(f'Coordenadas: x_start={x_start}, y_start={y_start}, '
                  f'x_end={x_end}, y_end={y_end}')

    image = image_name.copy()
    oriImage = image.copy()

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", mouse_crop_local)

    while True:
        i = image.copy()
        key = cv2.waitKey(2)
        
        if not cropping:
            cv2.imshow("image", image)
        else:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)

        if key % 256 == 27:
            cv2.destroyAllWindows()
            break
        
    return x_start, y_start, x_end, y_end

def data_norm(data):
    """Normalize data to [0, 1]"""
    return (data - data.min()) / (data.max() - data.min())

def smooth(Original, Original_Weight, Retrieved):
    """Smooth blending of two images"""
    Retrieved_Weight = 1 - Original_Weight
    return Original_Weight * Original + Retrieved_Weight * Retrieved

def smooth1(Original, Retrieved):
    """Default smooth blending (55% original)"""
    alpha = 0.55
    beta = 1 - alpha
    return alpha * Original + beta * Retrieved