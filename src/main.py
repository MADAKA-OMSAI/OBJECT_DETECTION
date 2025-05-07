import argparse
import os
from object_counting import process_video_and_count, process_image_and_count

def main():
    parser = argparse.ArgumentParser(description="Object Detection and Counting")
    parser.add_argument("--video_path", type=str, help="Path to the input video file")
    parser.add_argument("--image_path", type=str, help="Path to the input image file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLO model")
    parser.add_argument("--classes_to_count", type=int, nargs='+', required=True, help="List of class IDs to count")
    parser.add_argument("--run_dir", type=str, default="runs/temp", help="Directory to save output files")

    args = parser.parse_args()

    os.makedirs(args.run_dir, exist_ok=True)

    if args.video_path:
        print(f"Processing video: {args.video_path}")
        object_counts, output_video_path, input_video_path = process_video_and_count(
            args.video_path, args.model_path, args.classes_to_count, args.run_dir)
        print(f"Output video saved at: {output_video_path}")
        print(f"Object counts: {object_counts}")
    elif args.image_path:
        print(f"Processing image: {args.image_path}")
        object_counts, output_image_path = process_image_and_count(
            args.image_path, args.model_path, args.classes_to_count, args.run_dir)
        print(f"Output image saved at: {output_image_path}")
        print(f"Object counts: {object_counts}")
    else:
        print("Please provide either a video path or an image path.")

if __name__ == "__main__":
    main()
