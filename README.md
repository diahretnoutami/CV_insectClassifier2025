# 🐞 Insect Classifier - FastAPI

A web-based insect classification app using two TensorFlow models (CNN and MobileNet), deployed with FastAPI. Users can upload insect images and receive predictions from both models, along with accuracy and descriptions.

## 🧠 Models
- CNN model (`cnn_model.h5`)
- MobileNet model (`mobilenet_model.h5`)

## ⚙️ Tech Stack
- FastAPI 
- TensorFlow & Keras
- HTML/CSS (for frontend)
- Uvicorn (as ASGI server)
- Python
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn


## 📦 Dataset

This project uses the [Insects Recognition Dataset](https://www.kaggle.com/datasets/hammaadali/insects-recognition) by Hammaad Ali, available on Kaggle.

**Dataset Features:**
- Contains high-quality images of 5 different insect classes. There are grasshopper, butterfly, mosquito, ladybird and dragonfly.
- Organized into labeled folders for each class.
- Ideal for supervised image classification tasks.
- Image format: `.jpg`

The dataset was used to train both the CNN and MobileNet models included in this project.



## 🚀 How to Run Locally

Make sure you have all dependencies installed and your virtual environment activated.

**Step 1: Start the FastAPI backend**
Open a terminal and run:

```bash
venv\Scripts\activate
uvicorn app.main:app --reload
```

The backend will be running at:
http://127.0.0.1:8000

**Step 2: Start a local HTTP server (for the frontend)**
in another terminal, terminal, activate your virtual environment and run:

```bash
python -m http.server 8080
```
This will serve your frontend.html at:
http://127.0.0.1:8080/frontend.html

Now the frontend can communicate with the FastAPI backend.



