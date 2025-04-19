<h1>ECZEMA DETECTION PROJECT</h1>

A simple machine learning-powered web app to detect **eczema** from skin images.


<h4>📁 Dataset</h4>

*Download here:*
[Google Drive Dataset Link](https://drive.google.com/drive/folders/1hjXKMEu9vwkfCY4uuha3Ewb5ueQtQ01x?usp=drive_link)  
Contains labeled skin images organized into:
- `eczema/`
- `normal/`



<h4>How It Works</h4>

1. We trained a **Support Vector Machine (SVM)** model using labeled images of eczema and normal skin.
2. The model is deployed in a **Streamlit** web app for interactive image-based prediction.
3. Users upload a skin image.
4. The app preprocesses the image and predicts whether it's **eczema** or **normal**.
5. The result is displayed clearly on screen.

---

<h4>Project Structure</h4>

```
eczema_project/
│
├── dataset/              # Training data (eczema / normal images)
│   ├── eczema/
│   └── normal/
│
├── model_train.py        # Trains multiple ML models
├── eczema_model.pkl      # Saved trained model (SVM)
├── app.py                # Streamlit app for predictions
├── requirements.txt      # Project dependencies
└── .gitignore            # Version control exclusions
```



<h4>Models Used</h4>

We trained and tested multiple models to identify the best performer:

| Model                  | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| SVM                    | Best performance (~79.3% accuracy), used in deployed app                    |
| Random Forest          | Robust ensemble model, good with imbalanced data                            |
| K-Nearest Neighbors    | Simple baseline model, less effective for image data                        |
| MLP (Neural Net)       | Basic neural network with 2 hidden layers, decent performance               |
| CNN (Deep Learning)    | Custom CNN model trained with TensorFlow; limited due to hardware constraints |

*Only SVM is used in the current version of the app, but other models are available for experimentation.*

---

<h4> How to Run the Project</h4>

 1. Clone the repository
```bash
git clone <your-repository-url>
cd eczema_project
```

 2. (Optional) Create a virtual environment
```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

 3. Install dependencies
```bash
pip install -r requirements.txt
```

 4. (Optional) Train the model from scratch
```bash
python model_train.py
```
This will retrain all models and generate `eczema_model.pkl`.

 5. Run the Streamlit web app
```bash
streamlit run app.py
```

 6. Use the App
Open the URL (e.g. http://localhost:8501) in your browser and:
- Upload a skin image
- Get instant prediction
- Try with different images for testing

---

## ✅ Features

- 📷 Upload skin images to classify as **eczema** or **normal**
- ⚙️ Trained on real image data
- 💾 Multiple models tested, best selected
- 🌐 Simple and responsive **Streamlit** web interface

---

## 🔮 Future Improvements

- 🧠 Train deeper CNN models for better accuracy  
- 📈 Add support for real-time camera input  
- 🧬 Collect more diverse and larger dataset  
- 🧪 Test with pre-trained models (e.g., ResNet, MobileNet)  
- 🎨 Improve UI/UX of the web interface

---

## 🛠️ Tech Stack

- Python  
- Scikit-learn  
- Streamlit  
- NumPy  
- Pillow (PIL)  
- TensorFlow (for CNN training)

---

## 📂 Requirements

Make sure you have the following installed (or use the provided `requirements.txt`):

```txt
numpy
pillow
scikit-learn
streamlit
joblib
tensorflow
imblearn
matplotlib
```

---

## 💡 Credits
Supervisior : Dr. Amit Kumar

Created by a team of students exploring AI in medical imaging with limited resources — proving that even low-end hardware can still deliver smart solutions 💻❤️
```

Let me know if you also want a logo, a badge (like "Built with ❤️ using Streamlit"), or deployment instructions (e.g., on Hugging Face or Heroku)!
