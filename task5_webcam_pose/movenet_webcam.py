import time
import cv2
import numpy as np
from PIL import Image, ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

_NUM_KEYPOINTS = 17

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

EDGES = [
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle')
]

def draw_pose(img, pose, width, height):
    draw = ImageDraw.Draw(img)
    for i in range(_NUM_KEYPOINTS):
        y, x, score = pose[i]
        if score > 0.2:
            px = int(x * width)
            py = int(y * height)
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=(255, 0, 0))
    for edge in EDGES:
        start_idx = KEYPOINT_DICT[edge[0]]
        end_idx = KEYPOINT_DICT[edge[1]]
        y1, x1, s1 = pose[start_idx]
        y2, x2, s2 = pose[end_idx]
        if s1 > 0.2 and s2 > 0.2:
            draw.line((int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)), fill=(0, 255, 0), width=2)
    return img

def main():
    model = 'test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite'
    print('Initializing Edge TPU interpreter...')
    interpreter = make_interpreter(model)
    interpreter.allocate_tensors()
    input_size = common.input_size(interpreter)
    print(f'Input size: {input_size}')
    print("Opening webcam... Press 'q' to quit.")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened():
        print('Error: Could not open webcam.')
        return
    prev_time = time.time()
    fps = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Error: Failed to capture frame.')
            break
        frame = cv2.flip(frame, 1)
        orig_height, orig_width = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        resized_img = pil_img.resize(input_size, Image.LANCZOS)
        common.set_input(interpreter, resized_img)
        interpreter.invoke()
        pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
        annotated_img = resized_img.copy()
        annotated_img = draw_pose(annotated_img, pose, input_size[0], input_size[1])
        annotated_img = annotated_img.resize((orig_width, orig_height), Image.LANCZOS)
        result_cv = cv2.cvtColor(np.array(annotated_img), cv2.COLOR_RGB2BGR)
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time + 1e-6)
        prev_time = current_time
        cv2.putText(result_cv, f'FPS: {fps:.2f}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        right_wrist_idx = KEYPOINT_DICT['right_wrist']
        wrist_y, wrist_x, wrist_score = pose[right_wrist_idx]
        if wrist_score > 0.2:
            wrist_px = int(wrist_x * orig_width)
            wrist_py = int(wrist_y * orig_height)
            cv2.circle(result_cv, (wrist_px, wrist_py), 8, (0, 0, 255), -1)
            wrist_text = f'Right Wrist: ({wrist_px}, {wrist_py})'
            cv2.putText(result_cv, wrist_text, (10, orig_height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow('MoveNet Pose Detection - Task 5', result_cv)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Webcam session ended.')

if __name__ == '__main__':
    main()
