from pycoral.adapters import common
from pycoral.adapters import classify
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
from PIL import Image
import time

model = 'test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite'
labels = 'test_data/inat_bird_labels.txt'
input_image = 'test_data/parrot.jpg'

interpreter = make_interpreter(model)
interpreter.allocate_tensors()
size = common.input_size(interpreter)
image = Image.open(input_image).convert('RGB').resize(size, Image.LANCZOS)
common.set_input(interpreter, image)

for i in range(5):
    start = time.perf_counter()
    interpreter.invoke()
    classes = classify.get_classes(interpreter, top_k=1)
    inference_time = (time.perf_counter() - start) * 1000
    print(f'Inference {i+1}: {inference_time:.1f} ms')

label_map = read_label_file(labels)
for c in classes:
    print(label_map.get(c.id, c.id), c.score)
