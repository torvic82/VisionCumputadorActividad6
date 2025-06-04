import os
import uuid
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo
MODEL_PATH = 'Cow_classifier_python_final.h5'
model = load_model(MODEL_PATH)

# Diccionario de clases
class_names = {0: 'Ayshire', 1: 'Holstein', 2: 'Jersey', 3: 'Normando', 4: 'PardoSuizo'}

def prepare_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # cambia el tamaño si tu modelo requiere otro
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    csv_path = None

    if request.method == 'POST':
        files = request.files.getlist('images')
        session_id = str(uuid.uuid4())
        folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
        os.makedirs(folder, exist_ok=True)

        for file in files:
            if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                filename = secure_filename(file.filename)
                path = os.path.join(folder, filename)
                file.save(path)

                try:
                    img = prepare_image(path)
                    prediction = model.predict(img)
                    class_index = np.argmax(prediction)
                    class_name = class_names[class_index]
                    confidence = float(np.max(prediction))

                    results.append({
                        'filename': filename,
                        'predicted_class': class_name,
                        'confidence': round(confidence, 3)
                    })
                except Exception:
                    results.append({
                        'filename': filename,
                        'predicted_class': 'Error',
                        'confidence': 0.0
                    })

        # Guardar CSV
        df = pd.DataFrame(results)
        csv_path = os.path.join(folder, 'results.csv')
        df.to_csv(csv_path, index=False)

    return render_template('index.html', results=results, csv_path=csv_path)

@app.route('/download/<path:csv_path>')
def download_csv(csv_path):
    return send_file(csv_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
