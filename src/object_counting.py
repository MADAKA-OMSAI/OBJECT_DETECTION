import cv2
import json
import os
from collections import defaultdict
from ultralytics import YOLO

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

def count_objects(list1):
    count = {}
    for obj in list1:
        if obj not in count:
            count[obj] = 0
        count[obj] += 1
    return count

def count(set_obj):
    final_obj_list = []
    Obj_list = list(set_obj)
    for i in Obj_list:
        temp = i.split('_')
        final_obj_list.append(temp[0])
    return count_objects(final_obj_list)
def process_image_and_count(image_path, model_path, classes_to_count, run_dir, confidence_threshold=0.2):
    model = YOLO(model_path)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No image found at {image_path}")

    results = model.track(image, classes=classes_to_count, persist=True, conf=confidence_threshold, iou=0.6, stream=False)
    print(f"Detection results: {results}")

    object_counts = defaultdict(int) #default dict to count objects
    Final_obj = set()
    annotated_images = []

    for r in results:
        for box in r.boxes:
            if box.id is not None and box.cls[0] in classes_to_count:
                track_id = box.id.int().cpu().tolist()[0]
                class_id = int(box.cls[0])
                class_name = classNames[class_id]
                Final_obj.add(class_name + '_' + str(track_id)) #class name +track id

        annotated_image = r.plot()
        annotated_images.append(annotated_image) # bounding boxes and labels

    output_image_path = os.path.join(run_dir, os.path.basename(image_path))
    if annotated_images:
        cv2.imwrite(output_image_path, annotated_images[0])

    json_path = os.path.join(run_dir, "object_counts.json")
    with open(json_path, 'w') as f:
        json.dump(count(Final_obj), f, indent=4) # saves the counts of detected objects to a json file

    return count(Final_obj), output_image_path

def process_video_and_count(video_path, model_path, classes_to_count, run_dir, confidence_threshold=0.2):
    model = YOLO(model_path)
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise FileNotFoundError(f"No video found at {video_path}")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    output_video_path = os.path.join(run_dir, os.path.basename(video_path))
    output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    object_counts = defaultdict(int)
    Final_obj = set()

    while True:
        ret, frame = video.read()
        if not ret:
            break

        results = model.track(frame, classes=classes_to_count, persist=True, conf=confidence_threshold, iou=0.6, stream=False)
        
        for r in results:
            for box in r.boxes:
                if box.id is not None and box.cls[0] in classes_to_count:
                    track_id = box.id.int().cpu().tolist()[0]
                    class_id = int(box.cls[0])
                    class_name = classNames[class_id]
                    Final_obj.add(class_name + '_' + str(track_id))

            annotated_frame = r.plot()

        output_video.write(annotated_frame)

    video.release()
    output_video.release()

    json_path = os.path.join(run_dir, "object_counts.json")
    with open(json_path, 'w') as f:
        json.dump(count(Final_obj), f, indent=4)

    return count(Final_obj), output_video_path, video_path
