# debug_analyzer.py
from core_analyzer import process_video

if __name__ == "__main__":
    filepath = "sample_data/L6_point_five_six_feet_portrait.mov"
    # debug video only pops up on screen if there is no save_debug_path passed
    result = process_video(filepath, leg="left", draw_debug=True, save_debug_path="test.mp4", orientation="portrait")
    print(result)
