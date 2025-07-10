# debug_analyzer.py
from core_analyzer import process_video

if __name__ == "__main__":
    filepath = "Right_Knee.mov.mp4"
    result = process_video("Right_Knee.mov", leg="right", draw_debug=True)
    print(result)
