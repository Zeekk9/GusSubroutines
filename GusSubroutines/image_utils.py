import cv2
import numpy as np

# Variables globales para compatibilidad heredada
x_c, y_c = 0, 0

def mouse_crop(event, x, y, flags, param):
    """Callback global para capturar clics (compatibilidad)"""
    global x_c, y_c
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates of pixel: X: {x}, Y: {y}")
        x_c, y_c = x, y

def rescale(mat, new_min, new_max):
    """Normaliza y escala una matriz al rango [new_min, new_max]"""
    m_min, m_max = mat.min(), mat.max()
    if m_max == m_min: 
        return np.full(mat.shape, new_min)
    return (mat - m_min) / (m_max - m_min) * (new_max - new_min) + new_min

def _prepare_display_image(image, alpha=1.0, beta=1.0):
    """
    Ajusta el brillo y contraste de la imagen para visualización interactiva.
    - alpha: Controla el contraste (valores > 1 aumentan el contraste).
    - beta: Controla el brillo (valores positivos suman brillo, negativos oscurecen).
    """
    img_float = image.astype(np.float64)
    
    # Estiramiento base por percentiles para asegurar visibilidad inicial
    p_low, p_high = np.percentile(img_float, (1, 99))
    
    if p_high > p_low:
        scaled = np.clip((img_float - p_low) / (p_high - p_low), 0, 1) * 255
    else:
        scaled = rescale(img_float, 0, 255)
        
    # Aplicación de control manual de contraste (alpha) y brillo (beta)
    # Fórmula de OpenCV: g(x) = alpha * f(x) + beta
    adjusted = cv2.convertScaleAbs(scaled, alpha=alpha, beta=beta)
    
    # Si la imagen es en escala de grises, la convertimos a BGR para dibujar en color
    if len(adjusted.shape) == 2:
        disp_uint8 = cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)
    else:
        disp_uint8 = adjusted
        
    return disp_uint8

# =================================================================
# FUNCIONES INTERACTIVAS DE ROI Y SELECCIÓN
# =================================================================

def single_cord(image):
    """
    Selección interactiva de un punto único.
    """
    local_yc, local_xc = -1, -1
    has_selection = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal local_yc, local_xc, has_selection
        if event == cv2.EVENT_LBUTTONDOWN:
            local_yc, local_xc = y, x
            has_selection = True
            print(f"Punto seleccionado -> Y: {local_yc}, X: {local_xc}")

    # Usamos la versión con brillo mejorado para visualizar
    disp = _prepare_display_image(image)
    if len(disp.shape) == 2:
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    win_name = "Seleccion de Punto (Clic: seleccionar | ESC: confirmar)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback)

    while True:
        temp_img = disp.copy()
        if has_selection:
            cv2.drawMarker(temp_img, (local_xc, local_yc), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

        cv2.imshow(win_name, temp_img)
        key = cv2.waitKey(20) % 256

        if key == 27:  # ESC
            if has_selection:
                cv2.destroyWindow(win_name)
                break
            else:
                print("⚠️ Haz clic en la imagen antes de presionar ESC para confirmar.")

    cv2.destroyAllWindows()
    return local_yc, local_xc

def multi_cords(image, ancho, largo, n):
    """
    Selección interactiva de múltiples coordenadas de ROI.
    """
    cord = []
    display_image = _prepare_display_image(image)
    if len(display_image.shape) == 2:
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

    for i in range(n):
        local_yc, local_xc = -1, -1
        has_selection = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal local_yc, local_xc, has_selection
            if event == cv2.EVENT_LBUTTONDOWN:
                local_yc, local_xc = y, x
                has_selection = True
                print(f"ROI {i+1}/{n} - Coordenada elegida -> Y: {local_yc}, X: {local_xc}")

        win_name = f"Seleccion Multi-Cords {i+1}/{n} (Clic: ubicar | ESC: confirmar)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, mouse_callback)

        while True:
            temp_img = display_image.copy()
            if has_selection:
                y1, y2 = local_yc - 2 * ancho, local_yc
                x1, x2 = local_xc - largo, local_xc + largo
                cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow(win_name, temp_img)
            key = cv2.waitKey(20) % 256

            if key == 27:  # ESC
                if has_selection:
                    cord.append([local_yc, local_xc])
                    y1, y2 = local_yc - 2 * ancho, local_yc
                    x1, x2 = local_xc - largo, local_xc + largo
                    display_image[max(0, y1):min(display_image.shape[0], y2),
                                  max(0, x1):min(display_image.shape[1], x2)] = 0
                    print(f"✓ ROI {i+1} confirmado.")
                    cv2.destroyWindow(win_name)
                    break
                else:
                    print("⚠️ Haz clic en la imagen para generar el ROI antes de presionar ESC.")

    cv2.destroyAllWindows()
    return cord

def crop(image, ancho, largo, n):
    """
    Recorte interactivo de múltiples regiones.
    """
    Is = []
    cord = []
    cropped_base = image.copy()
    display_image = _prepare_display_image(image)
    if len(display_image.shape) == 2:
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

    for i in range(n):
        local_yc, local_xc = -1, -1
        has_selection = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal local_yc, local_xc, has_selection
            if event == cv2.EVENT_LBUTTONDOWN:
                local_yc, local_xc = y, x
                has_selection = True
                print(f"Recorte {i+1}/{n} -> Y: {local_yc}, X: {local_xc}")

        win_name = f"Recorte {i+1}/{n} (Clic: ubicar | ESC: confirmar)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, mouse_callback)

        while True:
            temp_img = display_image.copy()
            if has_selection:
                y1, y2 = local_yc - 2 * ancho, local_yc
                x1, x2 = local_xc - largo, local_xc + largo
                cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imshow(win_name, temp_img)
            key = cv2.waitKey(20) % 256

            if key == 27:  # ESC
                if has_selection:
                    y1, y2 = local_yc - 2 * ancho, local_yc
                    x1, x2 = local_xc - largo, local_xc + largo
                    
                    Is.append(cropped_base[max(0, y1):min(cropped_base.shape[0], y2),
                                           max(0, x1):min(cropped_base.shape[1], x2)])
                    cord.append([local_yc, local_xc])

                    display_image[max(0, y1):min(display_image.shape[0], y2),
                                  max(0, x1):min(display_image.shape[1], x2)] = 0
                    print(f"✓ Recorte {i+1} confirmado.")
                    cv2.destroyWindow(win_name)
                    break
                else:
                    print("⚠️ Haz clic en la imagen antes de presionar ESC para confirmar.")

    cv2.destroyAllWindows()
    return Is, cord

def multi_cords_center(image, largo, ancho, n):
    """
    Selección interactiva de múltiples regiones centradas en el clic.
    El primer ROI define Y; en los siguientes solo cambia X.
    """
    cord = []
    display_image = _prepare_display_image(image)
    if len(display_image.shape) == 2:
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

    fixed_y = None  # Almacena el centro en Y fijado en la primera selección

    for i in range(n):
        local_yc, local_xc = -1, -1
        has_selection = False

        def mouse_callback(event, x, y, flags, param):
            nonlocal local_yc, local_xc, has_selection
            if event == cv2.EVENT_LBUTTONDOWN:
                # Si ya se fijó fixed_y, usamos esa altura; de lo contrario, la del clic
                local_yc = fixed_y if fixed_y is not None else y
                local_xc = x
                has_selection = True
                print(f"ROI Centrado {i+1}/{n} -> Centro en Y={local_yc}, X={local_xc}")

        win_name = f"ROI Centrado {i+1}/{n} (Clic: ubicar | ESC: confirmar)"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, mouse_callback)

        while True:
            temp_img = display_image.copy()
            
            # Dibujar una guía horizontal en la coordenada Y fijada para facilitar la alineación
            if fixed_y is not None:
                cv2.line(temp_img, (0, fixed_y), (temp_img.shape[1], fixed_y), (255, 0, 0), 1)

            if has_selection:
                y1, y2 = local_yc - largo, local_yc + largo
                x1, x2 = local_xc - ancho, local_xc + ancho
                cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.circle(temp_img, (local_xc, local_yc), 3, (0, 0, 255), -1)

            cv2.imshow(win_name, temp_img)
            key = cv2.waitKey(20) % 256

            if key == 27:  # ESC
                if has_selection:
                    # Guardar el valor de Y en la primera iteración
                    if fixed_y is None:
                        fixed_y = local_yc
                        print(f"🔒 Altura Y fijada globalmente en: {fixed_y}")

                    cord.append([local_yc, local_xc])
                    
                    y1, y2 = local_yc - largo, local_yc + largo
                    x1, x2 = local_xc - ancho, local_xc + ancho
                    
                    display_image[max(0, y1):min(display_image.shape[0], y2), 
                                  max(0, x1):min(display_image.shape[1], x2)] = 0
                    print(f"✓ ROI Centrado {i+1} confirmado.")
                    cv2.destroyWindow(win_name)
                    break
                else:
                    print("⚠️ Haz clic en la imagen antes de presionar ESC para confirmar el centro.")

    cv2.destroyAllWindows()
    return cord

def single_crop(image, ancho, largo):
    """
    Recorte interactivo de una sola región.
    """
    cropped_base = image.copy()
    display_image = _prepare_display_image(image)
    if len(display_image.shape) == 2:
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)

    local_yc, local_xc = -1, -1
    has_selection = False

    def mouse_callback(event, x, y, flags, param):
        nonlocal local_yc, local_xc, has_selection
        if event == cv2.EVENT_LBUTTONDOWN:
            local_yc, local_xc = y, x
            has_selection = True
            print(f"Recorte único -> Y: {local_yc}, X: {local_xc}")

    win_name = "Recorte Unico (Clic: ubicar | ESC: confirmar)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback)

    while True:
        temp_img = display_image.copy()
        if has_selection:
            y1, y2 = local_yc - 2 * ancho, local_yc
            x1, x2 = local_xc - largo, local_xc + largo
            cv2.rectangle(temp_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(win_name, temp_img)
        key = cv2.waitKey(20) % 256

        if key == 27:  # ESC
            if has_selection:
                y1, y2 = local_yc - 2 * ancho, local_yc
                x1, x2 = local_xc - largo, local_xc + largo
                
                crop_result = cropped_base[max(0, y1):min(cropped_base.shape[0], y2),
                                           max(0, x1):min(cropped_base.shape[1], x2)]
                print("✓ Recorte único confirmado.")
                cv2.destroyWindow(win_name)
                break
            else:
                print("⚠️ Haz clic en la imagen antes de presionar ESC para confirmar.")

    cv2.destroyAllWindows()
    return crop_result

def ROI(image_input, max_display_size=800):
    """
    Selección interactiva de ROI libre rectangular mediante arrastre.
    """
    x_start, y_start, x_end, y_end = 0, 0, 0, 0
    cropping = False
    has_selection = False

    oriImage = image_input.copy()
    image = _prepare_display_image(image_input)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    aspect_ratio = w / h

    if w > h:
        win_w = max_display_size
        win_h = int(max_display_size / aspect_ratio)
    else:
        win_h = max_display_size
        win_w = int(max_display_size * aspect_ratio)

    def mouse_crop_local(event, x, y, flags, param):
        nonlocal x_start, y_start, x_end, y_end, cropping, has_selection

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

            if roi_w > 0 and roi_h > 0:
                has_selection = True
                roi = oriImage[y1:y2, x1:x2]
                
                roi_aspect = roi_w / roi_h
                crop_max_size = 500
                
                if roi_w > roi_h:
                    crop_win_w = crop_max_size
                    crop_win_h = int(crop_max_size / roi_aspect)
                else:
                    crop_win_h = crop_max_size
                    crop_win_w = int(crop_max_size * roi_aspect)

                cv2.namedWindow("Cropped Preview", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Cropped Preview", crop_win_w, crop_win_h)
                cv2.imshow("Cropped Preview", roi)

    win_name = "ROI Selection (Arrastre | ESC para confirmar)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_crop_local)
    cv2.resizeWindow(win_name, win_w, win_h)

    while True:
        i = image.copy()
        key = cv2.waitKey(20) % 256

        if cropping or has_selection:
            x1, x2 = min(x_start, x_end), max(x_start, x_end)
            y1, y2 = min(y_start, y_end), max(y_start, y_end)
            cv2.rectangle(i, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.imshow(win_name, i)
        else:
            cv2.imshow(win_name, image)

        if key == 27:  # ESC
            if has_selection:
                cv2.destroyAllWindows()
                break
            else:
                print("⚠️ Selecciona una región antes de presionar ESC para confirmar.")

    x_min, x_max = min(x_start, x_end), max(x_start, x_end)
    y_min, y_max = min(y_start, y_end), max(y_start, y_end)

    return x_min, y_min, x_max, y_max

def ROI_circular(image_input, max_display_size=800):
    """
    Selecciona un ROI circular mediante clic en el centro y arrastre.
    """
    x_center, y_center = 0, 0
    radius = 0
    selecting = False
    has_selection = False

    oriImage = image_input.copy()
    image = _prepare_display_image(image_input)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    h, w = image.shape[:2]
    aspect_ratio = w / h
    if w > h:
        win_w = max_display_size
        win_h = int(max_display_size / aspect_ratio)
    else:
        win_h = max_display_size
        win_w = int(max_display_size * aspect_ratio)

    img_disp = image.copy()

    def mouse_circle_local(event, x, y, flags, param):
        nonlocal x_center, y_center, radius, selecting, has_selection

        if event == cv2.EVENT_LBUTTONDOWN:
            x_center, y_center = x, y
            radius = 0
            selecting = True

        elif event == cv2.EVENT_MOUSEMOVE and selecting:
            radius = int(np.sqrt((x - x_center)**2 + (y - y_center)**2))

        elif event == cv2.EVENT_LBUTTONUP and selecting:
            radius = int(np.sqrt((x - x_center)**2 + (y - y_center)**2))
            selecting = False

            if radius > 0:
                has_selection = True
                x1 = max(0, x_center - radius)
                x2 = min(w, x_center + radius)
                y1 = max(0, y_center - radius)
                y2 = min(h, y_center + radius)

                crop_disp = img_disp[y1:y2, x1:x2].copy()
                grid_y, grid_x = np.ogrid[:crop_disp.shape[0], :crop_disp.shape[1]]
                c_y, c_x = y_center - y1, x_center - x1
                mask_preview = (grid_x - c_x)**2 + (grid_y - c_y)**2 <= radius**2

                crop_disp[~mask_preview] = 0

                cv2.namedWindow("Cropped Circular", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Cropped Circular", 500, 500)
                cv2.imshow("Cropped Circular", crop_disp)

    win_name = "ROI Circular (Clic+Arrastre | ESC: confirmar)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_circle_local)
    cv2.resizeWindow(win_name, win_w, win_h)

    while True:
        i = img_disp.copy()
        key = cv2.waitKey(20) % 256

        if (selecting or has_selection) and radius > 0:
            cv2.circle(i, (x_center, y_center), 3, (0, 0, 255), -1)
            cv2.circle(i, (x_center, y_center), radius, (255, 0, 0), 2)
            cv2.imshow(win_name, i)
        else:
            cv2.imshow(win_name, img_disp)

        if key == 27:  # ESC
            if has_selection and radius > 0:
                cv2.destroyAllWindows()
                break
            else:
                print("⚠️ Define un círculo antes de presionar ESC para confirmar.")

    x1 = max(0, x_center - radius)
    x2 = min(w, x_center + radius)
    y1 = max(0, y_center - radius)
    y2 = min(h, y_center + radius)

    crop = oriImage[y1:y2, x1:x2].copy()
    grid_y, grid_x = np.ogrid[:crop.shape[0], :crop.shape[1]]
    c_y, c_x = y_center - y1, x_center - x1
    mask = (grid_x - c_x)**2 + (grid_y - c_y)**2 <= radius**2

    crop[~mask] = np.nan

    return crop, mask, (y1, y2, x1, x2)

# =================================================================
# FUNCIONES AUXILIARES MATEMÁTICAS Y DE IMAGEN
# =================================================================

def data_norm(data):
    """Normaliza datos al rango [0, 1]"""
    return (data - data.min()) / (data.max() - data.min())

def smooth(Original, Original_Weight, Retrieved):
    """Mezcla suavizada de dos imágenes según pesos"""
    Retrieved_Weight = 1 - Original_Weight
    return Original_Weight * Original + Retrieved_Weight * Retrieved

def smooth1(Original, Retrieved):
    """Mezcla suavizada predeterminada (55% original)"""
    alpha = 0.55
    beta = 1 - alpha
    return alpha * Original + beta * Retrieved

def error_mask(matrix, error_percent=0, method='uniform'):
    """Genera una máscara de ruido estocástico"""
    if error_percent == 0:
        return np.zeros(np.shape(matrix))
    
    shape = np.shape(matrix)
    magnitude = np.mean(np.abs(matrix))
    
    if method == 'uniform':
        error_factor = np.random.uniform(-error_percent/100, error_percent/100, shape)
        noise = error_factor * magnitude
    elif method == 'normal':
        std = (error_percent/100) * magnitude
        noise = np.random.normal(0, std, shape)
    elif method == 'proportional':
        noise = (error_percent/100) * matrix * np.random.randn(*shape)
    elif method == 'relative':
        relative_factor = np.random.uniform(-error_percent/100, error_percent/100, shape)
        noise = relative_factor * np.abs(matrix)
    else:
        return matrix

    return noise

def apply_stochastic_noise(matrix, error_percent=0, method='uniform'):
    """Aplica ruido estocástico a arreglos (1D, 2D o escalares)"""
    if error_percent == 0:
        return matrix
    
    matrix = np.asanyarray(matrix)
    shape = matrix.shape
    magnitude = np.mean(np.abs(matrix))
    
    if method == 'uniform':
        noise = np.random.uniform(-error_percent/100, error_percent/100, size=shape) * magnitude
    elif method == 'normal':
        std = (error_percent/100) * magnitude
        noise = np.random.normal(0, std, size=shape)
    elif method == 'proportional':
        noise = (error_percent/100) * matrix * np.random.standard_normal(size=shape)
    elif method == 'relative':
        noise = np.random.uniform(-error_percent/100, error_percent/100, size=shape) * np.abs(matrix)
    else:
        return matrix

    return matrix + noise

def importing(image_path):
    """Carga imagen desde ruta y convierte a escala de grises float64"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en: {image_path}")
    
    image_float = image.astype(np.float64)
    if len(image_float.shape) == 3 and image_float.shape[2] == 3:
        return np.mean(image_float, axis=2)
        
    return image_float

def ensure_grayscale(image):
    """Asegura que la matriz de entrada sea escala de grises float64"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float64)
    return image.astype(np.float64)