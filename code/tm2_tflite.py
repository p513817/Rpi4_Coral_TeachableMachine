import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import re

def loadLabels(labelPath):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(labelPath, 'r', encoding='utf-8') as labelFile:
        lines = (p.match(line).groups() for line in labelFile.readlines())
        return {int(num): text.strip() for num, text in lines}

pkg = importlib.util.find_spec('tflite_runtime')

if pkg:
    print('tflite')
    from tflite_runtime.interpreter import Interpreter
else:
    print('tf.lite')
    from tensorflow.lite.python.interpreter import Interpreter

interpreter = Interpreter(model_path='tflite_model/model.tflite')
interpreter.allocate_tensors()

labels = loadLabels('tflite_model/labels.txt')

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

frame_rate_calc = 0
freq = cv2.getTickFrequency()

cap = cv2.VideoCapture(0)

while cap.isOpened():

    t1 = cv2.getTickCount()

    ret, frame = cap.read()

    scr = frame.copy()
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))

    # 加一個維度 補 batch size
    input_data = np.expand_dims(frame_resized, axis=0)

    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    interpreter.set_tensor( (input_details[0]['index']) , input_data)

    interpreter.invoke()

    results = interpreter.get_tensor( output_details[0]['index'])[0]

    print_res = f'FPS: {frame_rate_calc:.2f} , Label: {labels[np.argmax(results)]}'

    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    cv2.putText(frame, print_res, (30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
cap.release()
