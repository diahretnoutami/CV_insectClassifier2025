from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = FastAPI()

origins = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load 2 models
cnn_model_path = os.path.join(os.path.dirname(__file__), 'ProyekCV_model.h5')
mobilenet_model_path = os.path.join(os.path.dirname(__file__), 'ProyekCV_model_v2.h5')

cnn_model = load_model(cnn_model_path)
mobilenet_model = load_model(mobilenet_model_path)

class_names = ['Butterfly', 'Dragonfly', 'Grasshopper', 'Ladybird', 'Mosquito']

descriptions = {
    'Grasshopper': "Grasshopper adalah serangga herbivora yang dikenal dengan kemampuan melompat jauh berkat kaki belakangnya yang kuat. Mereka biasanya ditemukan di padang rumput atau lahan terbuka. Suara khas yang dihasilkan oleh grasshopper berasal dari gesekan sayap mereka. Hewan ini memegang peran penting dalam rantai makanan sebagai mangsa bagi berbagai predator. Dalam jumlah besar, beberapa spesies grasshopper dapat menjadi hama tanaman pertanian.",
    'Butterfly': "Butterfly adalah serangga cantik dengan sayap berwarna-warni yang hidup di berbagai habitat. Mereka mengalami metamorfosis lengkap dari larva menjadi dewasa. Butterfly sering dikaitkan dengan penyerbukan tanaman. Kehadiran mereka menandakan ekosistem yang sehat. Beberapa spesies butterfly terancam punah karena kerusakan habitat.",
    'Dragonfly': "Dragonfly adalah serangga pemangsa yang hidup di dekat air. Mereka terbang sangat cepat dan lincah, memakan serangga lain seperti nyamuk. Dragonfly memiliki mata besar yang memberikan penglihatan hampir 360 derajat. Mereka berperan penting dalam mengontrol populasi serangga hama. Larvanya hidup di air sebelum bermetamorfosis menjadi dewasa.",
    'Ladybird': "Ladybird, atau kepik, adalah serangga kecil berwarna cerah dengan bintik-bintik di punggungnya. Mereka dikenal sebagai predator alami kutu daun. Ladybird dianggap menguntungkan bagi petani karena membantu mengendalikan hama. Terdapat berbagai spesies ladybird dengan pola warna yang berbeda. Beberapa budaya menganggap ladybird sebagai pembawa keberuntungan.",
    'Mosquito': "Mosquito adalah serangga kecil yang dikenal sebagai penghisap darah. Beberapa spesies dapat menularkan penyakit seperti malaria dan dengue. Hanya nyamuk betina yang menggigit manusia untuk mendapatkan protein dari darah. Mereka berkembang biak di air tergenang. Pengendalian populasi nyamuk penting untuk kesehatan masyarakat."
}

def preprocess_image(image):
    img = image.resize((150, 150))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img = preprocess_image(image)

    # CNN
    cnn_pred = cnn_model.predict(img)
    cnn_class = int(np.argmax(cnn_pred[0]))
    cnn_conf = float(np.max(cnn_pred[0]))

    if cnn_conf < 0.5:
        cnn_label = "Tidak Dikenali"
        cnn_desc = "Gambar tidak dapat dikenali dengan tingkat kepercayaan yang memadai."
    else:
        cnn_label = class_names[cnn_class]
        cnn_desc = descriptions.get(cnn_label, "Deskripsi tidak tersedia.")

    # MobileNet
    mobile_pred = mobilenet_model.predict(img)
    mobile_class = int(np.argmax(mobile_pred[0]))
    mobile_conf = float(np.max(mobile_pred[0]))

    if mobile_conf < 0.5:
        mobile_label = "Tidak Dikenali"
        mobile_desc = "Gambar tidak dapat dikenali dengan tingkat kepercayaan yang memadai."
    else:
        mobile_label = class_names[mobile_class]
        mobile_desc = descriptions.get(mobile_label, "Deskripsi tidak tersedia.")

    return JSONResponse(content={
        "cnn": {
            "predicted_class": cnn_label,
            "confidence": cnn_conf,
            "description": cnn_desc
        },
        "mobilenet": {
            "predicted_class": mobile_label,
            "confidence": mobile_conf,
            "description": mobile_desc
        }
    })

