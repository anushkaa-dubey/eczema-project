import os
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib
from collections import Counter

# Function to load and augment images
def load_images_from_folder(folder, label, augment=False):
    images = []
    labels = []
    
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((64, 64))
            img_array = np.array(img).flatten()
            
            images.append(img_array)
            labels.append(label)
            
            if augment:
                # Flip
                flipped = ImageOps.mirror(img)
                images.append(np.array(flipped).flatten())
                labels.append(label)

                # Rotate
                rotated = img.rotate(15)
                images.append(np.array(rotated).flatten())
                labels.append(label)

                # Invert colors
                inverted = ImageOps.invert(img)
                images.append(np.array(inverted).flatten())
                labels.append(label)

        except Exception as e:
            print(f"Error loading image {filename}: {e}")

    return images, labels

# Load eczema images
eczema_images, eczema_labels = load_images_from_folder('dataset/eczema', label=1)

# Load normal images with augmentation ✅
normal_images, normal_labels = load_images_from_folder('dataset/normal', label=0, augment=True)

# Combine both datasets
X = np.array(eczema_images + normal_images)
y = np.array(eczema_labels + normal_labels)

# Sanity check: print how many images in each category
print("Label Distribution:", Counter(y))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
model = SVC(kernel='linear', probability=True)
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print(f"Test Accuracy: {score * 100:.2f}%")

# Save the model
joblib.dump(model, 'eczema_model.pkl')
print("Model saved as eczema_model.pkl")
