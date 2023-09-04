import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
from ttkthemes import ThemedStyle

class VehicleDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detecção de Veículos")

        style = ThemedStyle(root)
        style.set_theme("arc")

        self.cap = None  # VideoCapture object
        self.net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

        with open('coco.names', 'r') as f:
            self.classes = f.read().strip().split('\n')

        self.car_class_id = self.classes.index('car')
        self.motobike_class_id = self.classes.index('motorbike')

        self.layer_names = self.net.getLayerNames()
        self.output_layer_names = self.net.getUnconnectedOutLayersNames()

        self.output_layer_indices = [self.layer_names.index(i) for i in self.output_layer_names]
        self.output_layers = [self.layer_names[i] for i in self.output_layer_indices]

        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.button_frame = ttk.Frame(root)
        self.button_frame.pack(padx=10, pady=10, anchor=tk.CENTER)

        self.start_button = ttk.Button(self.button_frame, text="Iniciar Detecção", command=self.toggle_detection)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.open_file_button = ttk.Button(self.button_frame, text="Abrir Arquivo de Vídeo", command=self.open_video_file)
        self.open_file_button.pack(side=tk.LEFT, padx=5)

        self.photo = None
        self.is_detecting = False
        self.process_video()

    def open_video_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4")])
        if file_path:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            self.is_detecting = False
            self.start_button.config(text="Iniciar Detecção")

    def toggle_detection(self):
        self.is_detecting = not self.is_detecting
        if self.is_detecting:
            self.start_button.config(text="Parar Detecção")
        else:
            self.start_button.config(text="Iniciar Detecção")

    def process_video(self):
        if self.is_detecting and self.cap is not None:
            ret, frame = self.cap.read()

            if ret:
                blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(608, 608), swapRB=True, crop=False, ddepth=cv2.CV_32F, mean=(0, 0, 0))
                self.net.setInput(blob)
                outs = self.net.forward(self.output_layers)

                class_ids = []
                confidences = []
                boxes = []

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]

                        if confidence > 0.5 and (class_id == self.car_class_id or class_id == self.motobike_class_id):
                            center_x = int(detection[0] * frame.shape[1])
                            center_y = int(detection[1] * frame.shape[0])
                            w = int(detection[2] * frame.shape[1])
                            h = int(detection[3] * frame.shape[0])

                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)

                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

                for i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    confidence = confidences[i]
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.root.after(10, self.process_video)

root = tk.Tk()
app = VehicleDetectionApp(root)
root.mainloop()
