CENG260 Lab L03 – Coral AI USB Accelerator
==========================================

Overview
This repository contains all source code and configuration used for CENG260 – Smart Embedded Systems, Lab L03: Coral AI USB Accelerator, completed on a Raspberry Pi 4 with a Coral USB Accelerator and USB webcam. The lab walks through installing the Edge TPU runtime and PyCoral, running an image classification model (MobileNet V2), performing single‑image human pose estimation with MoveNet, and extending MoveNet to a live webcam stream with FPS and right‑wrist tracking.

The code is organized per task so it is easy to rerun or extend for future projects (for example, a bicep‑curl counter using pose landmarks).

Hardware and Software

Raspberry Pi 4 running Raspberry Pi OS Bullseye (Python 3.9).

Coral USB Accelerator connected via USB 3.0.

HD Webcam eMeet C960 connected to a Pi USB port (/dev/video0).

Packages:

libedgetpu1-std (Edge TPU runtime)

python3-tflite-runtime

python3-pycoral

python3-opencv (OpenCV for webcam capture and display)

Model and test data are the same as Coral’s official examples:

mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite, inat_bird_labels.txt, parrot.jpg for classification.

movenet_single_pose_lightning_ptq_edgetpu.tflite, squat.bmp for pose estimation.

Repository Structure
CENG260-Lab03-Coral-AI/
├── README.md
├── requirements_notes.md
├── setup/
│ ├── install_coral_runtime.sh
│ └── install_pycoral.sh
├── task3_classification/
│ ├── classify_image_hardcoded.py
│ └── run_task3.sh
├── task4_pose/
│ ├── movenet_pose_estimation_hardcoded.py
│ └── run_task4.sh
├── task5_webcam_pose/
│ ├── movenet_webcam.py
│ └── run_task5.sh
└── screenshots/
├── task1_installation.png
├── task2_pycoral_lsusb.png
├── task3_classifier_output.png
├── task4_results_bmp.png
└── task5_webcam_output.png

setup/: Shell scripts to install the Edge TPU runtime and PyCoral from Coral’s Debian repository.

task3_classification/: Code to run MobileNet V2 bird classification on the Edge TPU (hardcoded paths, based on Coral’s classify_image.py).

task4_pose/: Code to run MoveNet pose estimation on a squat image with hardcoded model/input/output paths (adapted from Coral’s movenet_pose_estimation.py).

task5_webcam_pose/: Real‑time MoveNet pose tracking on a USB webcam stream with FPS overlay and right‑wrist annotation.

screenshots/: PNG/JPEG images captured during the lab for the written report.

Setup Instructions
On the Raspberry Pi:

git clone https://github.com/<your-username>/CENG260-Lab03-Coral-AI.git
cd CENG260-Lab03-Coral-AI

Install Edge TPU runtime + TFLite runtime
bash setup/install_coral_runtime.sh

Install PyCoral
bash setup/install_pycoral.sh

Install OpenCV (Debian package)
sudo apt install -y python3-opencv

These steps follow Coral’s “Get started with the USB Accelerator” flow for Debian/Raspberry Pi and the PyCoral installation instructions.

Task 3 – Image Classification on Edge TPU
Folder: task3_classification/

Script: classify_image_hardcoded.py

Loads mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite via PyCoral.

Resizes parrot.jpg to the expected input size.

Runs multiple inferences and prints per‑inference timing plus the top bird label and confidence (Scarlet Macaw).

Runner: run_task3.sh

Calls Coral’s examples/install_requirements.sh classify_image.py to download test data and then runs the hardcoded script.

Run:

cd task3_classification
bash run_task3.sh

Add task3_classifier_output.png in screenshots/ showing the inference times and predicted class.

Task 4 – MoveNet Pose Estimation (Single Image)
Folder: task4_pose/

Script: movenet_pose_estimation_hardcoded.py

Based on Coral’s MoveNet example but replaces argparse with fixed paths for the MoveNet Lightning Edge TPU model and squat.bmp.

Runs inference once, prints the 17×3 keypoint array, and writes Results.bmp with skeleton overlay.

Runner: run_task4.sh

Ensures MoveNet test data is downloaded and runs the script.

Run:

cd task4_pose
bash run_task4.sh

Add task4_results_bmp.png showing the squat pose detection.

Task 5 – Webcam Pose Detection with FPS and Right Wrist Annotation
Folder: task5_webcam_pose/

Script: movenet_webcam.py

Loads movenet_single_pose_lightning_ptq_edgetpu.tflite on the Edge TPU via PyCoral.

Opens the USB webcam with cv2.VideoCapture(0, cv2.CAP_V4L2) (HD Webcam eMeet C960 on /dev/video0).

For each frame:

Flips horizontally, converts to RGB, resizes to 192×192, and runs MoveNet.

Draws the 17‑keypoint skeleton on the resized frame and rescales it back to full resolution.

Computes FPS using frame‑to‑frame timestamps and overlays “FPS: XX.XX” at top‑left.

Extracts the right wrist keypoint (index 10), draws a large red circle at its pixel location, and overlays “Right Wrist: (x, y)” at bottom‑left.

Displays live output in a window titled “MoveNet Pose Detection – Task 5”, quit with q.

Run:

cd task5_webcam_pose
bash run_task5.sh

Take a screenshot of the running window and save it as screenshots/task5_webcam_output.png (for example, the one showing ~9.6 FPS and wrist coordinates). Press q to close.

How to Use This Repo for the Lab Report

Source code appendix: Point to the GitHub repo and briefly explain that all scripts are split per task.

Screenshots: Attach images from screenshots/ in the report sections for Tasks 1–5.

Future work: Task 5’s movenet_webcam.py can be extended to compute distances between shoulder–elbow–wrist keypoints and implement a bicep‑curl repetition counter, building on the same MoveNet output used here.
