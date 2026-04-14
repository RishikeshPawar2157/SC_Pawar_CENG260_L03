from PIL import Image, ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

_NUM_KEYPOINTS = 17

KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

EDGES = [
    ('nose', 'left_eye'), ('nose', 'right_eye'),
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'), ('left_hip', 'left_knee'),
    ('left_knee', 'left_ankle'), ('right_hip', 'right_knee'),
    ('right_knee', 'right_ankle')
]

def draw_pose(img, pose, width, height):
    draw = ImageDraw.Draw(img)
    for i in range(_NUM_KEYPOINTS):
        if pose[i][2] > 0.2:
            draw.ellipse([
                pose[i][1] * width - 3, pose[i][0] * height - 3,
                pose[i][1] * width + 3, pose[i][0] * height + 3
            ], fill=(255, 0, 0))
    for edge in EDGES:
        s = KEYPOINT_DICT[edge[0]]
        e = KEYPOINT_DICT[edge[1]]
        if pose[s][2] > 0.2 and pose[e][2] > 0.2:
            draw.line([
                pose[s][1] * width, pose[s][0] * height,
                pose[e][1] * width, pose[e][0] * height
            ], fill=(0, 255, 0), width=2)
    return img

model = 'test_data/movenet_single_pose_lightning_ptq_edgetpu.tflite'
input_file = 'test_data/squat.bmp'
output_file = 'Results.bmp'

interpreter = make_interpreter(model)
interpreter.allocate_tensors()
input_size = common.input_size(interpreter)
img = Image.open(input_file).convert('RGB')
resized_img = img.resize(input_size, Image.LANCZOS)
common.set_input(interpreter, resized_img)
interpreter.invoke()
pose = common.output_tensor(interpreter, 0).copy().reshape(_NUM_KEYPOINTS, 3)
result = draw_pose(img.copy(), pose, img.width, img.height)
result.save(output_file)
print(pose)
print('Saved to', output_file)
