import cv2
import os
import shutil

def video_to_frames(video_path, jpg_quality=85, scale=1.0):
    """
    Decomposes a video into frames optimized for file size.
    
    Args:
        video_path (str): Path to the input video file (e.g., .avi, .mp4).
        jpg_quality (int): JPEG quality (1-100, 85 recommended).
        scale (float): Scaling factor (e.g., 0.5 for 50% smaller dimensions).
    """
    
    if not os.path.exists(video_path):
        print(f"Error: File '{video_path}' not found.")
        return
    
    # Use the filename (without extension) as the output directory
    output_dir = os.path.splitext(video_path)[0]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Directory '{output_dir}' created.")

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {total_frames} frames @ {fps:.1f} FPS")
    #print(f'\rCurrent frame: frame_{i}.jpg', end='', flush=True)
    
    frame_count = 0
    total_size = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # OPTION 1: Resize frame if scale is not 1.0
        if scale != 1.0:
            new_width = int(frame.shape[1] * scale)
            new_height = int(frame.shape[0] * scale)
            frame = cv2.resize(frame, (new_width, new_height), 
                             interpolation=cv2.INTER_AREA)

        # OPTION 2: Convert to grayscale (optional, significantly reduces size)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # OPTION 3: Save as JPG with compression
        frame_name = f"frame_{frame_count}.jpg" # Added padding for better sorting
        full_path = os.path.join(output_dir, frame_name)
        
        # JPEG compression parameters
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpg_quality]
        cv2.imwrite(full_path, frame, encode_params)
        
        # Track file size
        file_size = os.path.getsize(full_path)
        total_size += file_size
        
        frame_count += 1
        
        if frame_count % 50 == 0:
        # El \r al inicio borra visualmente la línea anterior al sobreescribirla
            print(f"\rProcessed {frame_count}/{total_frames} frames...", end='', flush=True)

    cap.release()
    
    avg_size = total_size / frame_count if frame_count > 0 else 0
    print(f"\n✅ Completed: {frame_count} frames")
    print(f"📦 Total size: {total_size/1024/1024:.2f} MB")
    print(f"📊 Average size per frame: {avg_size/1024:.2f} KB")


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