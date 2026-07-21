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

def single_cord(image):
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

def multi_cords(image, ancho, largo, n):
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

        '''Is.append(np.mean(cv2.fastNlMeansDenoising(
            image[y_c-2*ancho:y_c, x_c-largo:x_c+largo]), axis=2) * 1.0)'''
        cord.append([y_c, x_c])
        image[y_c-2*ancho:y_c, x_c-largo:x_c+largo] = 0

    return cord

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

        '''Is.append(np.mean(cv2.fastNlMeansDenoising(
            image[y_c-2*ancho:y_c, x_c-largo:x_c+largo]), axis=2) * 1.0)'''
        Is.append(croped[y_c-2*ancho:y_c, x_c-largo:x_c+largo])
        cord.append([y_c, x_c])
        image[y_c-2*ancho:y_c, x_c-largo:x_c+largo] = 0

    return Is, cord

def single_crop(image, ancho, largo):
    """Interactive cropping of multiple regions"""
    Is = []
    cord = []
    alpha = 1  # Contrast control
    beta = 1   # Brightness control
    croped = image
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", mouse_crop)

    while True:
        key = cv2.waitKey(2)
        cv2.imshow("image", image)
        
        if key % 256 == 27:
            cv2.destroyAllWindows()
            break

        '''Is.append(np.mean(cv2.fastNlMeansDenoising(
            image[y_c-2*ancho:y_c, x_c-largo:x_c+largo]), axis=2) * 1.0)'''
    Is=croped[y_c-2*ancho:y_c, x_c-largo:x_c+largo]
    image[y_c-2*ancho:y_c, x_c-largo:x_c+largo] = 0
    
    while True:
            key = cv2.waitKey(2)
            cv2.imshow("image", image)
            
            if key % 256 == 27:
                cv2.destroyAllWindows()
                break

    return Is

import cv2

import cv2

def ROI(image_input, max_display_size=800):
    """
    Permite seleccionar una Región de Interés (ROI) manteniendo la relación de aspecto
    tanto en la imagen principal como en la vista previa del recorte.
    
    :param image_input: Imagen de entrada (Matriz NumPy)
    :param max_display_size: Tamaño máximo en píxeles (ancho o alto) para la ventana
    """
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    cropping = False
    
    oriImage = image_input.copy()
    image = image_input.copy()

    # 1. Calcular tamaño de la ventana principal manteniendo ASPECT RATIO
    h, w = image.shape[:2]
    aspect_ratio = w / h

    if w > h:
        win_w = max_display_size
        win_h = int(max_display_size / aspect_ratio)
    else:
        win_h = max_display_size
        win_w = int(max_display_size * aspect_ratio)

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

            x1, x2 = min(x_start, x_end), max(x_start, x_end)
            y1, y2 = min(y_start, y_end), max(y_start, y_end)

            roi_w = x2 - x1
            roi_h = y2 - y1

            print(f'Coordenadas ROI: x_start={x1}, y_start={y1}, x_end={x2}, y_end={y2}')

            # Si hay una selección válida, calculamos el aspect ratio de la ROI
            if roi_w > 0 and roi_h > 0:
                roi = oriImage[y1:y2, x1:x2]
                
                # 2. Redimensionar la ventana "Cropped" manteniendo SU propio aspect ratio
                roi_aspect = roi_w / roi_h
                crop_max_size = 500  # Límite de tamaño para la ventana emergente
                
                if roi_w > roi_h:
                    crop_win_w = crop_max_size
                    crop_win_h = int(crop_max_size / roi_aspect)
                else:
                    crop_win_h = crop_max_size
                    crop_win_w = int(crop_max_size * roi_aspect)

                cv2.namedWindow("Cropped", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Cropped", crop_win_w, crop_win_h)
                cv2.imshow("Cropped", roi)

    # Configuración de la ventana principal
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", mouse_crop_local)
    cv2.resizeWindow('image', win_w, win_h)  # Redimensionado proporcional

    while True:
        i = image.copy()
        key = cv2.waitKey(2)
        
        if not cropping:
            cv2.imshow("image", image)
        else:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)
            cv2.imshow("image", i)

        if key % 256 == 27:  # Salir con ESC
            cv2.destroyAllWindows()
            break

    x_min, x_max = min(x_start, x_end), max(x_start, x_end)
    y_min, y_max = min(y_start, y_end), max(y_start, y_end)

    return x_min, y_min, x_max, y_max

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
    Aplica ruido estocástico a cualquier arreglo (1D, 2D o escalar).
    Actualizada para evitar IndexError en perfiles 1D.
    """
    # 1. Caso base: si no hay error, devolvemos original
    if error_percent == 0:
        return matrix
    
    # Aseguramos que sea un array de numpy para manejar las dimensiones correctamente
    matrix = np.asanyarray(matrix)
    shape = matrix.shape
    
    # Magnitud media para escalar errores
    magnitude = np.mean(np.abs(matrix))
    
    # 2. Generar el ruido usando el argumento 'size' (evita el error de índices)
    if method == 'uniform':
        noise = np.random.uniform(-error_percent/100, error_percent/100, size=shape) * magnitude
    
    elif method == 'normal':
        std = (error_percent/100) * magnitude
        noise = np.random.normal(0, std, size=shape)
    
    elif method == 'proportional':
        # Reemplazamos np.random.randn(*shape) por standard_normal(size=shape)
        noise = (error_percent/100) * matrix * np.random.standard_normal(size=shape)
    
    elif method == 'relative':
        noise = np.random.uniform(-error_percent/100, error_percent/100, size=shape) * np.abs(matrix)
    
    else:
        return matrix

    # 3. Retornar la matriz original más el ruido
    return matrix + noise

def importing(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en: {image_path}")
    
    image_float = image.astype(np.float64) / 255.0  # Normaliza a [0.0, 1.0]
    
    if len(image_float.shape) == 3 and image_float.shape[2] == 3:
        return np.mean(image_float, axis=2)
        
    return image_float

def ensure_grayscale(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Convertimos a gris y luego a float para preservar precisión numérica
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float64)
    return image.astype(np.float64)

def ROI_circular(image_input, max_display_size=800):
    """
    Selecciona un ROI circular mediante clic en el centro y arrastre para definir el radio.
    Muestra la vista previa con fondo negro dentro de la ROI circular.
    
    :param image_input: Matriz de la imagen (float64 o uint8).
    :param max_display_size: Tamaño máximo de ventana conservando Aspect Ratio.
    :return: (crop, mask, coords)
             - crop: Imagen recortada (fuera del círculo = np.nan).
             - mask: Máscara booleana (True dentro del círculo).
             - coords: (y1, y2, x1, x2) coordenadas en la imagen original.
    """
    x_center, y_center = 0, 0
    radius = 0
    selecting = False
    
    oriImage = image_input.copy()
    image = image_input.copy()

    # Redimensionado proporcional de la ventana principal
    h, w = image.shape[:2]
    aspect_ratio = w / h
    if w > h:
        win_w = max_display_size
        win_h = int(max_display_size / aspect_ratio)
    else:
        win_h = max_display_size
        win_w = int(max_display_size * aspect_ratio)

    # Preparar copia para visualización en OpenCV (Normalizada a uint8 de 0-255)
    img_disp = image.copy()
    if img_disp.dtype != np.uint8:
        # Asegurar rango 0-255 para visualización limpia
        img_disp = cv2.normalize(img_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def mouse_circle_local(event, x, y, flags, param):
        nonlocal x_center, y_center, radius, selecting
        
        # 1. Clic izquierdo: Define el centro
        if event == cv2.EVENT_LBUTTONDOWN:
            x_center, y_center = x, y
            radius = 0
            selecting = True

        # 2. Arrastre: Define el radio dinámico
        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            radius = int(np.sqrt((x - x_center)**2 + (y - y_center)**2))

        # 3. Soltar clic: Finaliza la selección y muestra el recorte circular
        elif event == cv2.EVENT_LBUTTONUP and selecting:
            radius = int(np.sqrt((x - x_center)**2 + (y - y_center)**2))
            selecting = False

            if radius > 0:
                # Bounding box del círculo
                x1 = max(0, x_center - radius)
                x2 = min(w, x_center + radius)
                y1 = max(0, y_center - radius)
                y2 = min(h, y_center + radius)

                # Extraer para la vista previa desde la versión uint8
                crop_disp = img_disp[y1:y2, x1:x2].copy()
                
                # Crear máscara circular
                grid_y, grid_x = np.ogrid[:crop_disp.shape[0], :crop_disp.shape[1]]
                c_y, c_x = y_center - y1, x_center - x1
                mask_preview = (grid_x - c_x)**2 + (grid_y - c_y)**2 <= radius**2
                
                # Poner en 0 (negro) lo que esté fuera del radio en la vista previa
                crop_disp[~mask_preview] = 0

                cv2.namedWindow("Cropped Circular", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Cropped Circular", 500, 500)
                cv2.imshow("Cropped Circular", crop_disp)

    # Ventana principal
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("image", mouse_circle_local)
    cv2.resizeWindow('image', win_w, win_h)

    # Convertir img_disp a BGR solo para dibujar la guía azul/roja
    img_bgr = cv2.cvtColor(img_disp, cv2.COLOR_GRAY2BGR) if len(img_disp.shape) == 2 else img_disp.copy()

    while True:
        i = img_bgr.copy()
        key = cv2.waitKey(2)
        
        if selecting and radius > 0:
            # Dibujar punto central rojo y circunferencia azul
            cv2.circle(i, (x_center, y_center), 3, (0, 0, 255), -1)
            cv2.circle(i, (x_center, y_center), radius, (255, 0, 0), 2)
            cv2.imshow("image", i)
        else:
            cv2.imshow("image", img_bgr)

        if key % 256 == 27:  # Salir con tecla ESC
            cv2.destroyAllWindows()
            break

    # Construir el recorte final con la precisión matemática original (float64)
    x1 = max(0, x_center - radius)
    x2 = min(w, x_center + radius)
    y1 = max(0, y_center - radius)
    y2 = min(h, y_center + radius)

    crop = oriImage[y1:y2, x1:x2].copy()
    grid_y, grid_x = np.ogrid[:crop.shape[0], :crop.shape[1]]
    c_y, c_x = y_center - y1, x_center - x1
    mask = (grid_x - c_x)**2 + (grid_y - c_y)**2 <= radius**2

    # Asignar NaN a la región exterior para Matplotlib / VES
    crop[~mask] = np.nan

    return crop, mask, (y1, y2, x1, x2)
