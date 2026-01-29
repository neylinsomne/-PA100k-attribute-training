"""
Test de atributos humanos con CPU
Prueba el modelo actual de atributos sin GPU
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

# Configurar ONNX Runtime para usar solo CPU
ort.set_default_logger_severity(3)  # Solo errores

# Paths
WEIGHTS_DIR = Path(__file__).parent / "output_pytorch"
HUMAN_ATTR_MODEL = WEIGHTS_DIR / "human_attr_pytorch.onnx"
VIDEO_PATH = Path(__file__).parent / "test_videos" / "attributes_sim.mp4"

# Atributos (27 atributos - incluye Male)
ATTRIBUTES = [
    "Female", "AgeOver60", "Age18-60", "AgeLess18",
    "Front", "Side", "Back",
    "Hat", "Glasses",
    "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
    "ShortSleeve", "LongSleeve",
    "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
    "LowerStripe", "LowerPattern",
    "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots",
    "Male"  # Atributo 27
]

def preprocess_image(image, target_size=(192, 256)):
    """Preprocess image for attribute model"""
    # Resize
    img = cv2.resize(image, target_size)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std

    # HWC to CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    return img.astype(np.float32)

def detect_persons_simple(frame):
    """Detector simple de personas usando HOG"""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detectar personas
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    return boxes

def predict_attributes(session, image):
    """Predecir atributos de una imagen"""
    # Preprocess
    input_tensor = preprocess_image(image)

    # Inference
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})[0]

    # Sigmoid activation
    probs = 1 / (1 + np.exp(-outputs[0]))

    # Threshold
    predictions = (probs > 0.5).astype(int)

    return predictions, probs

def main():
    print("=" * 70)
    print("Test de Atributos Humanos - CPU Only")
    print("=" * 70)

    # Verificar archivos
    print(f"\nModelo: {HUMAN_ATTR_MODEL}")
    if not HUMAN_ATTR_MODEL.exists():
        print(f"ERROR: Modelo no encontrado!")
        return

    print(f"Video: {VIDEO_PATH}")
    if not VIDEO_PATH.exists():
        print(f"ERROR: Video no encontrado!")
        return

    # Cargar modelo en CPU
    print("\nCargando modelo ONNX (CPU)...")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(HUMAN_ATTR_MODEL),
        sess_options,
        providers=['CPUExecutionProvider']
    )
    print(f"Modelo cargado: {session.get_providers()}")

    # Abrir video
    print(f"\nAbriendo video...")
    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("ERROR: No se pudo abrir el video")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}, Total frames: {total_frames}")
    print(f"Duración: {total_frames/fps:.1f}s")

    # Procesar frames (cada 10 frames para velocidad)
    frame_idx = 0
    detections = []

    print("\nProcesando frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Procesar solo cada 10 frames
        if frame_idx % 10 != 0:
            continue

        # Detectar personas
        boxes = detect_persons_simple(frame)

        if len(boxes) == 0:
            continue

        print(f"\nFrame {frame_idx}: {len(boxes)} persona(s) detectada(s)")

        # Para cada persona detectada
        for i, (x, y, w, h) in enumerate(boxes):
            # Recortar región de la persona
            person_img = frame[y:y+h, x:x+w]

            if person_img.size == 0:
                continue

            # Predecir atributos
            predictions, probs = predict_attributes(session, person_img)

            # Mostrar atributos detectados
            print(f"  Persona {i+1}:")

            # Género (Female=0, Male=26)
            female_prob = probs[0]
            male_prob = probs[26] if len(probs) > 26 else 1 - female_prob
            is_female = female_prob > male_prob
            print(f"    Género: {'Mujer' if is_female else 'Hombre'} (F:{female_prob:.2f} M:{male_prob:.2f})")

            # Edad
            if predictions[1]:
                print(f"    Edad: >60 años ({probs[1]:.2f})")
            elif predictions[2]:
                print(f"    Edad: 18-60 años ({probs[2]:.2f})")
            elif predictions[3]:
                print(f"    Edad: <18 años ({probs[3]:.2f})")

            # Bolsos
            bags = []
            if predictions[9]: bags.append(f"HandBag({probs[9]:.2f})")
            if predictions[10]: bags.append(f"ShoulderBag({probs[10]:.2f})")
            if predictions[11]: bags.append(f"Backpack({probs[11]:.2f})")
            if bags:
                print(f"    Bolsos: {', '.join(bags)}")

            # Otros atributos activos
            active_attrs = []
            for idx, (pred, prob) in enumerate(zip(predictions, probs)):
                if pred == 1 and idx not in [0, 1, 2, 3, 9, 10, 11]:  # Excluir ya mostrados
                    active_attrs.append(f"{ATTRIBUTES[idx]}({prob:.2f})")

            if active_attrs:
                print(f"    Otros: {', '.join(active_attrs[:5])}")  # Max 5

            detections.append({
                'frame': frame_idx,
                'person': i+1,
                'gender': 'Female' if is_female else 'Male',
                'female_prob': float(female_prob),
                'male_prob': float(male_prob),
                'attributes': {ATTRIBUTES[idx]: float(prob) for idx, pred in enumerate(predictions) if pred == 1}
            })

    cap.release()

    print("\n" + "=" * 70)
    print(f"Procesamiento completo")
    print(f"Total detecciones: {len(detections)}")
    print("=" * 70)

    # Resumen
    print("\nResumen:")
    female_count = sum(1 for d in detections if d['gender'] == 'Female')
    male_count = sum(1 for d in detections if d['gender'] == 'Male')
    print(f"  Mujeres detectadas: {female_count}")
    print(f"  Hombres detectados: {male_count}")

    print("\nNOTA: El modelo fine-tuned (human_attr_pytorch.onnx) tiene 27 atributos.")
    print("      Incluye 'Male' como atributo separado (índice 26).")
    print("      Ambos Female y Male son predichos independientemente.")

if __name__ == "__main__":
    main()
