import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model

# Inicializa Flask
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar modelo
MODEL_PATH = 'Cow_classifier_python_final.h5'
model = load_model(MODEL_PATH)

# Definición de clases
class_names = ['Ayshire', 'Holstein', 'Jersey', 'Normando', 'PardoSuizo']

# Ruta principal
@app.route('/')
def index():
    return render_template('index.html')

# Ruta para predecir
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Procesar imagen
        img = Image.open(file_path).resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Hacer predicción
        predictions = model.predict(img_array)[0]
        results = list(zip(class_names, predictions))
        results.sort(key=lambda x: x[1], reverse=True)

        return render_template('index.html', predictions=results, filename=filename)

if __name__ == '__main__':
    # Crear carpeta si no existe
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
