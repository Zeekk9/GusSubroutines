import cv2
import os
import shutil

def video_to_frames(video_path, output_path=None, jpg_quality=85, scale=1.0):
    """
    Decomposes a video into frames.
    
    Args:
        video_path (str): Path to the input video file.
        output_path (str): Custom directory to save frames. If None, uses video filename.
        jpg_quality (int): JPEG quality (1-100).
        scale (float): Scaling factor.
    """
    import os
    import cv2

    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        return
    
    # Si output_path es None, usa el nombre del video en la raíz actual
    # Si es una ruta (ej: 'C:/MisFrames'), usará esa.
    if output_path is None:
        output_dir = os.path.splitext(os.path.basename(video_path))[0]
    else:
        output_dir = output_path
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    total_size = 0
    
    print(f"Processing: {total_frames} frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if scale != 1.0:
            new_width = int(frame.shape[1] * scale)
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Usamos :05d para que los nombres sean frame_00001.jpg, esto evita errores de orden
        frame_name = f"frame_{frame_count:05d}.jpg" 
        full_path = os.path.join(output_dir, frame_name)
        
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
        cv2.imwrite(full_path, frame, encode_params)
        
        total_size += os.path.getsize(full_path)
        frame_count += 1
        
        if frame_count % 50 == 0:
            print(f"\rProcessed {frame_count}/{total_frames} frames...", end='', flush=True)

    cap.release()
    print(f"\n✅ Completed: {frame_count} frames in '{output_dir}'")
    print(f"📦 Total size: {total_size/1024/1024:.2f} MB")


def count_jpg_files(directory):
    """
    Counts and lists JPG files in a given directory.

    Args:
        directory (str): Path to the directory to search.

    Returns:
        int: Number of JPG files found.
    """
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return 0
    
    files = sorted(os.listdir(directory))
    jpg_list = [f for f in files if f.lower().endswith(".jpg")]
    
    print(f"Number of JPGs: {len(jpg_list)}")
    return len(jpg_list)

def delete_temporal_files(video):
    """
    Delete the directory and all the files
    """
    output_dir = os.path.splitext(video)[0]
    try:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"✓ Path '{output_dir}' y all files are deleted")
        else:
            print(f"✗ The path '{output_dir}' does not exist")
    except Exception as e:
        print(f"Error to delete temporal files: {e}")



# --- USAGE EXAMPLES ---

# 1. Normal compression
# video_to_frames("input_video.avi", jpg_quality=85)

# 2. High compression & Scaling down
# video_to_frames("input_video.avi", jpg_quality=60, scale=0.5)
