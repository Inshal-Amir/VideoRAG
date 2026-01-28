import cv2
import base64
import os
from moviepy import VideoFileClip

class VideoProcessor:
    @staticmethod
    def get_frame_base64(image):
        """Encodes an OpenCV image to Base64 string."""
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')

    def extract_frames(self, video_path, interval=1):
        """
        Generator that yields frames from the video.
        Yields: (timestamp, base64_image)
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Safety check for invalid videos
        if fps == 0:
            return

        stride = int(fps * interval)
        current_frame = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if current_frame % stride == 0:
                timestamp = current_frame / fps
                # Resize to 640x360 to reduce API costs and latency
                frame = cv2.resize(frame, (640, 360))
                b64_img = self.get_frame_base64(frame)
                
                yield timestamp, b64_img
                
            current_frame += 1
        
        cap.release()

    def create_clip(self, video_path, start_time, end_time, output_path):
        """
        Cuts the video file to the exact start/end times safely.
        """
        try:
            # Check if file exists to prevent errors
            if not os.path.exists(video_path):
                print(f"Error: Video file not found at {video_path}")
                return False

            # Use VideoFileClip (Context Manager ensures file is closed properly)
            with VideoFileClip(video_path) as video:
                # Ensure we don't cut past the end of the video
                end = min(video.duration, end_time)
                start = max(0, start_time)
                
                # Check if the clip is valid (length > 0)
                if start >= end:
                    print("Error: Start time is after end time.")
                    return False

                # COMPATIBILITY FIX: MoviePy 2.0 renamed .subclip() to .subclipped()
                if hasattr(video, 'subclipped'):
                    new_clip = video.subclipped(start, end)
                else:
                    new_clip = video.subclip(start, end)
                
                # Write the file to disk
                new_clip.write_videofile(
                    output_path, 
                    codec="libx264", 
                    audio_codec="aac", 
                    preset="ultrafast",
                    logger=None 
                )
            
            return True
            
        except Exception as e:
            print(f"Error creating clip: {e}")
            return False