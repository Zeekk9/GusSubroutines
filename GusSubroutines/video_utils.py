import cv2
import os

def video_to_frames(nombre_video, calidad_jpg=85, escala=1.0):
    """
    Descompone un video en frames optimizados para tamaño.
    
    Args:
        nombre_video (str): Ruta al archivo de video .avi
        calidad_jpg (int): Calidad JPEG (1-100, 85 recomendado)
        escala (float): Factor de escala (0.5 = 50% más pequeño)
    """
    
    if not os.path.exists(nombre_video):
        print(f"Error: El archivo '{nombre_video}' no existe.")
        return
    
    directorio_salida = os.path.splitext(nombre_video)[0]
    
    if not os.path.exists(directorio_salida):
        os.makedirs(directorio_salida)
        print(f"Directorio '{directorio_salida}' creado.")

    cap = cv2.VideoCapture(nombre_video)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir '{nombre_video}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {fps:.1f} FPS")
    
    frame_count = 0
    total_size = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔽 OPCIÓN 1: Redimensionar si se solicita
        if escala != 1.0:
            new_width = int(frame.shape[1] * escala)
            new_height = int(frame.shape[0] * escala)
            frame = cv2.resize(frame, (new_width, new_height), 
                             interpolation=cv2.INTER_AREA)

        # 🔽 OPCIÓN 2: Convertir a escala de grises (75% más ligero)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 🔽 OPCIÓN 3: Guardar como JPG con compresión
        nombre_frame = f"I_{frame_count}.jpg"
        ruta_completa = os.path.join(directorio_salida, nombre_frame)
        
        # Parámetros de compresión JPEG
        params = [cv2.IMWRITE_JPEG_QUALITY, calidad_jpg]
        cv2.imwrite(ruta_completa, frame, params)
        
        # Calcular tamaño del archivo
        file_size = os.path.getsize(ruta_completa)
        total_size += file_size
        
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"Procesados {frame_count}/{total_frames} frames...")

    cap.release()
    
    avg_size = total_size / frame_count if frame_count > 0 else 0
    print(f"\n✅ Completado: {frame_count} frames")
    print(f"📦 Tamaño total: {total_size/1024/1024:.1f} MB")
    print(f"📊 Tamaño promedio por frame: {avg_size/1024:.1f} KB")

# --- EJEMPLOS DE USO ---

# Opción 1: Compresión normal (recomendado)
# descomponer_video_en_frames_optimizado("image_21.avi", calidad_jpg=85)

# Opción 2: Máxima compresión
# descomponer_video_en_frames_optimizado("image_21.avi", calidad_jpg=60)

# Opción 3: Reducir tamaño + compresión
# descomponer_video_en_frames_optimizado("image_21.avi", calidad_jpg=80, escala=0.7)


def contar_archivos_png(directorio):
    if not os.path.isdir(directorio):
        print(f"Error: El directorio '{directorio}' no existe.")
        return 0, []
    
    archivos = sorted(os.listdir(directorio))
    lista_jpgs = [f for f in archivos if f.endswith(".jpg")]
    print("N of jpgs:", len(lista_jpgs))
    return len(lista_jpgs)