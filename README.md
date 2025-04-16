ECZEMA PROJECT


A simple machine learning-powered web app to detect **eczema** from skin images.

---
## DATA SET DRIVE LINK
https://drive.google.com/drive/folders/1hjXKMEu9vwkfCY4uuha3Ewb5ueQtQ01x?usp=drive_link

## 🚀 How it Works
- We trained a **Support Vector Machine (SVM)** model on labeled images of **eczema** and **normal** skin.
- Users can upload an image through a **Streamlit** web interface.
- The app preprocesses the image and predicts whether the skin is **eczema-affected** or **normal**.
- It shows the result on screen with a simple message.

---

## 🗂️ Project Structure

```
eczema_project/
│
├── dataset/              # Training data (eczema / normal images)
│   ├── eczema/
│   └── normal/
│
├── model_train.py        # Trains the machine learning model (SVM)
├── eczema_model.pkl      # Saved trained model
├── app.py                # Streamlit app for user prediction
├── requirements.txt      # Python dependencies
└── .gitignore            # Files & folders to ignore in version control
```

---

## ⚙️ How to Run the Project

### 1. Clone the repository
```bash
git clone <your-repository-url>
cd eczema_project
```

### 2. Set up a virtual environment (optional but recommended)
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. (Optional) Train the model from scratch
> If you want to retrain the model with your own dataset:
```bash
python model_train.py
```
This will create a file called `eczema_model.pkl`.

### 5. Run the Streamlit app
```bash
streamlit run app.py
```

### 6. Open the app in your browser
- After running the above command, Streamlit will give you a **local URL** (e.g., `http://localhost:8501`).
- Open it in your browser, upload an image, and see the prediction.

---

## ✅ Features
- Classifies skin images as **eczema** or **normal**.
- Easy-to-use web interface.
- Model trained on sample dataset (works best with more data!).

---

## ✅ Future Improvements
- Use a **Convolutional Neural Network (CNN)** for better accuracy.
- Collect a **larger and more diverse dataset**.
- Improve the UI/UX of the web app.
- Add **real-time camera input** support.

---

## 🛠️ Tech Stack
- Python
- Scikit-learn
- Streamlit
- NumPy
- Pillow (PIL)

---

## 📂 Requirements
Make sure you have the following installed (or install them via `pip install -r requirements.txt`):
- numpy
- pillow
- scikit-learn
- streamlit
- joblib

---

---



