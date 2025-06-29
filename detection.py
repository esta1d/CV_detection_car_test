import cv2
from ultralytics import YOLO
from shapely.geometry import Polygon, Point, box
import numpy as np

MODEL_PATH = '/home/estaid/dev/prediction_Car/yolov8x.pt'
VIDEO_INPUT_PATH = '/home/estaid/dev/prediction_Car/data/cvtest.avi'
VIDEO_OUTPUT_PATH = 'output_video.mp4'

# ROI
TOP_LEFT = (0, 440)
BOTTOM_LEFT = (0, 1514)
BOTTOM_RIGHT = (1677, 1514)
TOP_RIGHT = (1677, 440)
ROI_POLYGON = Polygon([TOP_LEFT, BOTTOM_LEFT, BOTTOM_RIGHT, TOP_RIGHT])

MODEL = YOLO(MODEL_PATH)

# классы для детекции
CLASSES = (2, 7)

def draw_filtered_boxes(frame, filtered_results):
    """
    Функция рисует рамочки и надписи поверх кадра для фильтрованных результатов.
    :param frame: исходный кадр
    :param filtered_results: список объектов, прошедших фильтрацию
    """
    for r in filtered_results:
        x1, y1, x2, y2 = map(int, r[:4])
        conf = round(float(r[4]), 2)
        cls = int(r[5])
        
        color = (0, 255, 0)  #green
        thickness = 3
        font_scale = 2
        text_color = (0, 0, 255) #red
        
        # Рисуем рамочку вокруг объекта
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Пишем метку класса и уверенность
        label = f"{cls}: {conf}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def process_frame(frame):
    """
    Обрабатываем один кадр: выполняем детекцию, фильтруем и рисуем обнаруженные объекты.
    :param frame: входящий кадр
    :return: кадр с нарисованными объектами
    """
    global MODEL, CLASSES
    
    results = MODEL.predict(frame, classes=CLASSES)[0]
    
    # Фильтрация объектов по региону интереса
    filtered_results = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = map(float, r[:6])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # Проверяем центр рамки внутри ROI
        if ROI_POLYGON.contains(Point(center_x, center_y)):
            filtered_results.append(r)
    
    # Отображаем ограничивающие рамки и метки на кадре
    draw_filtered_boxes(frame, filtered_results)
    
    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    
    # Получаем характеристики видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Создаем объект VideoWriter для вывода результата
    out = cv2.VideoWriter(
        VIDEO_OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = process_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()