import cv2
import numpy as np

# Abre o arquivo de vídeo
cap = cv2.VideoCapture('cars.mp4')

# Carregar o modelo YOLO
net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

# Carregar os nomes das classes
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().strip().split('\n')

# Definir as classes que serão detectadas, no caso carros, motos e caminhões
car_class_id = classes.index('car')
motobike_class_id = classes.index('motorbike')
truck_class_id = classes.index('truck')

# Definir as camadas de saída da rede YOLO
layer_names = net.getLayerNames()
output_layer_names = net.getUnconnectedOutLayersNames()

output_layer_indices = [layer_names.index(i) for i in output_layer_names]
output_layers = [layer_names[i] for i in output_layer_indices]

# Loop para processar cada quadro do vídeo
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    height, width, channels = frame.shape

    # Pré-processamento do quadro para a entrada da rede YOLO
    blob = cv2.dnn.blobFromImage(frame, scalefactor=0.00392, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Processamento das saídas da rede para detecção de veículos
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and class_id == car_class_id or class_id == motobike_class_id or class_id == truck_class_id:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordenadas para desenhar o retângulo
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar a supressão não máxima para evitar detecções redundantes
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            print(x, y, w, h)
            print(classes[class_ids[i]], confidences[i])
            label = str(classes[class_ids[i]] + ' ' + str(round(confidences[i], 2)))
            confidence = confidences[i]
            color = (0, 255, 0)  # Cor do retângulo (verde)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Detecção de Veículos', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
