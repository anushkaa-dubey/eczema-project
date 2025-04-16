import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib

def load_images_from_folder(folder, label, img_size=(32, 32)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)
            img_array = np.array(img).flatten()
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
    print(labels)
    return images, labels

# Load datasets
eczema_images, eczema_labels = load_images_from_folder('dataset/eczema', label=1)
normal_images, normal_labels = load_images_from_folder('dataset/normal', label=0)

# Combine datasets
X = np.array(eczema_images + normal_images)
y = np.array(eczema_labels + normal_labels)

print(f"Original dataset size: {len(X)} images")

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"After SMOTE resampling: {len(X_resampled)} images")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_predict = model.predict(X_test)
print(y_predict.shape)
print(y_test.shape)
# Evaluate model
score = model.score(y_predict, y_test)
print(f"Test Accuracy: {score * 100:.2f}%")

# Save model
joblib.dump(model, 'eczema_model.pkl')
print("Model saved as eczema_model.pkl")
