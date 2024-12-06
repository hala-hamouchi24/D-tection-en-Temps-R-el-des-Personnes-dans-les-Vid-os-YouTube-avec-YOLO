import cv2
import numpy as np
from pytube import YouTube
import os

# Fonction de téléchargement de la vidéo depuis YouTube
def download_youtube_video(video_url, save_path="."):
    yt = YouTube(video_url)
    stream = yt.streams.filter(file_extension='mp4', progressive=True).first()
    if not stream:
        raise Exception("No suitable streams found")
    stream.download(output_path=save_path)
    return os.path.join(save_path, stream.default_filename)

# Fonction de chargement du modèle YOLOv4-tiny
def load_yolo_model(weights_path, config_path, names_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Le fichier '{weights_path}' n'existe pas.")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Le fichier '{config_path}' n'existe pas.")
    if not os.path.exists(names_path):
        raise FileNotFoundError(f"Le fichier '{names_path}' n'existe pas.")

    net = cv2.dnn.readNet(weights_path, config_path)
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]
    return net, classes

# Fonction de détection des personnes dans une frame
def detect_people_in_frame(net, output_layers, frame):
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    people_coords = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Classe 'person'
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                people_coords.append((x, y, w, h))
    return people_coords

# Fonction principale de détection des personnes dans une vidéo
def detect_people_in_video(video_url):
    video_path = download_youtube_video(video_url)

    weights_path = r"C:\Users\dell\OneDrive\Bureau\tpATI2\yolov4-tiny.weights"
    config_path = r"C:\Users\dell\OneDrive\Bureau\tpATI2\yolov4-tiny (3).cfg"
    names_path = r"C:\Users\dell\OneDrive\Bureau\tpATI2\coco.names"

    net, classes = load_yolo_model(weights_path, config_path, names_path)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        people_coords = detect_people_in_frame(net, output_layers, frame)

        for x, y, w, h in people_coords:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        people_count = len(people_coords)
        cv2.putText(frame, f'Personnes detectees: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# URL de la vidéo YouTube
video_url = "https://www.youtube.com/watch?v=aa1oj3zC8Uo"

# Détecter les personnes marchant dans la vidéo YouTube
detect_people_in_video(video_url)
