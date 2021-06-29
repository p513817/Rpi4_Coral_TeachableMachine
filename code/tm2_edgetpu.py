from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
import cv2
import re
import os

# the TFLite converted to be used with edgetpu
modelPath = 'edgetpu_model/model_edgetpu.tflite'

# The path to labels.txt that was downloaded with your model
labelPath = 'edgetpu_model/labels.txt'

# This function parses the labels.txt and puts it in a python dictionary
def loadLabels(labelPath):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(labelPath, 'r', encoding='utf-8') as labelFile:
        lines = (p.match(line).groups() for line in labelFile.readlines())
        return {int(num): text.strip() for num, text in lines}

# This function takes in a PIL Image from any source or path you choose
def classifyImage(image_path, engine):
    # Load and format your image for use with TM2 model
    # image is reformated to a square to match training
    # image = Image.open(image_path)
    image = image_path
    image.resize((224, 224))

    # Classify and ouptut inference
    classifications = engine.classify_with_image(image)
    return classifications

def main():

    # Load your model onto your Coral Edgetpu
    engine = ClassificationEngine(modelPath)
    labels = loadLabels(labelPath)
    
    # 取得 CPU的頻率，為了取得最準確的fps，最後要除以cpu的頻率才正確
    freq = cv2.getTickFrequency()
    # 用於儲存FPS的變數
    frame_rate_calc = 0

    # 取得攝影機物件
    cap = cv2.VideoCapture(0)
    
    # 如果攝影機開啟
    while cap.isOpened():
        # 先儲存一開始的時間
        t1 = cv2.getTickCount()
        # 取得畫面，如果沒有畫面就跳出while
        ret, frame = cap.read()
        if not ret:
            break

        # 轉成 PIL 格式，通常神經網路模型都是吃 PIL 格式
        cv2_im = frame.copy()
        pil_im = Image.fromarray(cv2_im)

        # 縮放到模型輸入大小，並且水平反轉與訓練時相同
        pil_im.resize((224, 224))
        pil_im.transpose(Image.FLIP_LEFT_RIGHT)

        # 進行辨識，最後獲得結果
        results = classifyImage(pil_im, engine)[0]

        # 遇顯示在影像上的內容
        print_res = f'FPS: {frame_rate_calc:.2f} , Label: {labels[results[0]]}'

        # 將 FPS 畫在影像上並顯示
        cv2.putText(frame, print_res, (30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        cv2.imshow('frame', frame)

        # 計算 FPS (framerate)
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc= 1/time1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
