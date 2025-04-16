import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
import joblib
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Constants
IMG_SIZE = (128, 128)  
BATCH_SIZE = 32
EPOCHS = 2  #for faster training, later we will change it to 30

def load_images_from_folder(folder, label, img_size=IMG_SIZE):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0  # Normalize pixel values
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Failed to load {img_path}: {e}")
    return np.array(images), np.array(labels)

def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    # Flatten images for traditional ML models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # Dictionary to store models and their performance
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'),
        "SVM": SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
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
        
        # Save the model
        joblib.dump(model, f'{name.lower().replace(" ", "_")}_model.pkl')
    
    return results

def train_cnn(X_train, X_test, y_train, y_test):
    print("\nTraining CNN Model...")
    input_shape = X_train.shape[1:]
    
    # Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    model = create_cnn_model(input_shape)
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Eczema'])
    
    print(f"CNN Accuracy: {accuracy:.4f}")
    print(report)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.savefig('training_history.png')
    plt.close()
    
    # Save the model
    model.save('cnn_model.h5')
    
    return {
        "model": model,
        "accuracy": accuracy,
        "report": report
    }

def main():
    # Load datasets
    eczema_images, eczema_labels = load_images_from_folder('dataset/eczema', label=1)
    normal_images, normal_labels = load_images_from_folder('dataset/normal', label=0)

    # Combine datasets
    X = np.concatenate((eczema_images, normal_images))
    y = np.concatenate((eczema_labels, normal_labels))

    print(f"\nDataset Info:")
    print(f"Total images: {len(X)}")
    print(f"Eczema images: {sum(y)}")
    print(f"Normal images: {len(y) - sum(y)}")
    print(f"Image shape: {X[0].shape}")

    # Split dataset (before resampling to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Handle imbalance with SMOTE (only on training data)
    # For CNN, we'll use class weights instead
    smote = SMOTE(random_state=42)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_flat, y_train)
    X_train_resampled = X_train_resampled.reshape(-1, *X_train.shape[1:])

    print("\nAfter SMOTE resampling:")
    print(f"Training set size: {len(X_train_resampled)}")
    print(f"Eczema samples: {sum(y_train_resampled)}")
    print(f"Normal samples: {len(y_train_resampled) - sum(y_train_resampled)}")

    # Train and evaluate traditional ML models
    print("\nTraining traditional ML models...")
    ml_results = train_and_evaluate_models(X_train_resampled, X_test, y_train_resampled, y_test)

    # Train and evaluate CNN
    cnn_result = train_cnn(X_train, X_test, y_train, y_test)

    # Save all results
    all_results = {
        "traditional_ml": ml_results,
        "cnn": cnn_result
    }
    joblib.dump(all_results, 'model_results.pkl')

    # Print best model
    best_model_name = max(ml_results, key=lambda x: ml_results[x]['accuracy'])
    best_acc = ml_results[best_model_name]['accuracy']
    
    if cnn_result['accuracy'] > best_acc:
        print("\nBest model: CNN with accuracy {:.2f}%".format(cnn_result['accuracy'] * 100))
    else:
        print("\nBest traditional model: {} with accuracy {:.2f}%".format(
            best_model_name, best_acc * 100))

if __name__ == "__main__":
    main()