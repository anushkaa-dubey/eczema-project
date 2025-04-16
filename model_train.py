import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import joblib
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Constants (optimized for speed/performance balance)
IMG_SIZE = (64, 64)  # Reduced from 128x128 for faster training
BATCH_SIZE = 32
EPOCHS = 5           # Kept low for quick training (originally 2)

def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                img = img.convert('RGB').resize(IMG_SIZE)
                img_array = np.array(img) / 255.0  # Normalize
                images.append(img_array)
                labels.append(label)
        except Exception as e:
            print(f"Skipped {filename}: {str(e)}")
    return np.array(images), np.array(labels)

def create_improved_cnn_model():
    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 3)),  # Proper input layer
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),  # Reduced overfitting
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D(),  # Better than Flatten
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    models_dict = {
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        "SVM": SVC(kernel='rbf', C=10, probability=True, class_weight='balanced'),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    }
    
    results = {}
    for name, model in models_dict.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_flat, y_train)
        y_pred = model.predict(X_test_flat)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Normal', 'Eczema'])
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "report": report
        }
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(report)
        joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')
    
    return results

def train_cnn(X_train, X_test, y_train, y_test):
    print("\nTraining Improved CNN Model...")
    
    # Calculate class weights
    weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(weights))
    
    model = create_improved_cnn_model()
    
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        class_weight=class_weights,
        verbose=1
    )
    
    # Evaluation
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Eczema'])
    
    print(f"CNN Accuracy: {accuracy:.4f}")
    print(report)
    
    # Save model in modern Keras format
    model.save('cnn_model.keras')  # Instead of .h5
    
    return {
        "model": model,
        "accuracy": accuracy,
        "report": report
    }

def main():
    # Load datasets
    eczema_images, eczema_labels = load_images_from_folder('dataset/eczema', 1)
    normal_images, normal_labels = load_images_from_folder('dataset/normal', 0)

    # Combine and split
    X = np.concatenate((eczema_images, normal_images))
    y = np.concatenate((eczema_labels, normal_labels))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # Handle imbalance with SMOTE (for traditional models)
    smote = SMOTE()
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)
    X_train_resampled = X_train_resampled.reshape(-1, *IMG_SIZE, 3)

    # Train models
    print("\n=== Training Traditional Models ===")
    ml_results = train_and_evaluate_models(X_train_resampled, X_test, y_train_resampled, y_test)
    
    print("\n=== Training CNN ===")
    cnn_result = train_cnn(X_train, X_test, y_train, y_test)

    # Final comparison
    best_model = max(ml_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest Model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.2%}")

if __name__ == "__main__":
    main()