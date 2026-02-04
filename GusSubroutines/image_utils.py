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

def rescale(mat, new_min, new_max):
            m_min, m_max = mat.min(), mat.max()
            if m_max == m_min: return np.full(mat.shape, new_min) # Evitar división por cero
            # Normaliza a [0, 1] y luego escala a [new_min, new_max]
            return (mat - m_min) / (m_max - m_min) * (new_max - new_min) + new_min

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


def error_mask(matrix, error_percent=0, method='uniform'):
    """
    Aplica ruido estocástico a cualquier matriz (coeficientes, amplitudes, etc.)
    Retorna la matriz con ruido o la original si error_percent es 0.
    """
    # 1. Si no hay error, devolvemos la matriz vacia
    if error_percent == 0:
        return np.zeros(np.shape(matrix))
    
    shape = np.shape(matrix)
    # Magnitud media para escalar errores que no son proporcionales punto a punto
    magnitude = np.mean(np.abs(matrix))
    
    # 2. Generar el factor de error según el método
    if method == 'uniform':
        # Ruido uniforme basado en un porcentaje del valor promedio
        error_factor = np.random.uniform(-error_percent/100, error_percent/100, shape)
        noise = error_factor * magnitude
    
    elif method == 'normal':
        # Ruido gaussiano (normal) con desviación estándar proporcional al porcentaje
        std = (error_percent/100) * magnitude
        noise = np.random.normal(0, std, shape)
    
    elif method == 'proportional':
        # El ruido es más fuerte donde la señal es más alta (multiplicativo)
        # Esto es ideal para ruido de disparo (shot noise)
        noise = (error_percent/100) * matrix * np.random.randn(*shape)
    
    elif method == 'relative':
        # Ruido uniforme relativo al valor absoluto de cada píxel
        relative_factor = np.random.uniform(-error_percent/100, error_percent/100, shape)
        noise = relative_factor * np.abs(matrix)
    
    else:
        return matrix

    # 3. Retornar ruido generado
    return noise


def apply_stochastic_noise(matrix, error_percent=0, method='uniform'):
    """
    Aplica ruido estocástico a cualquier matriz (coeficientes, amplitudes, etc.)
    Retorna la matriz con ruido o la original si error_percent es 0.
    """
    # 1. Si no hay error, devolvemos la matriz original intacta
    if error_percent == 0:
        return matrix
    
    shape = np.shape(matrix)
    # Magnitud media para escalar errores que no son proporcionales punto a punto
    magnitude = np.mean(np.abs(matrix))
    
    # 2. Generar el factor de error según el método
    if method == 'uniform':
        # Ruido uniforme basado en un porcentaje del valor promedio
        error_factor = np.random.uniform(-error_percent/100, error_percent/100, shape)
        noise = error_factor * magnitude
    
    elif method == 'normal':
        # Ruido gaussiano (normal) con desviación estándar proporcional al porcentaje
        std = (error_percent/100) * magnitude
        noise = np.random.normal(0, std, shape)
    
    elif method == 'proportional':
        # El ruido es más fuerte donde la señal es más alta (multiplicativo)
        # Esto es ideal para ruido de disparo (shot noise)
        noise = (error_percent/100) * matrix * np.random.randn(*shape)
    
    elif method == 'relative':
        # Ruido uniforme relativo al valor absoluto de cada píxel
        relative_factor = np.random.uniform(-error_percent/100, error_percent/100, shape)
        noise = relative_factor * np.abs(matrix)
    
    else:
        return matrix

    # 3. Retornar la matriz original más el ruido generado
    return matrix + noise