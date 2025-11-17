from preprocessing.dataset_loader import DatasetLoader
from preprocessing.feature_extractor import prepare_training_data, normalize_and_split
from model.cnn_model import build_cnn_model

# 1. Load data
dataset = DatasetLoader().load_dataset(limit=10)

# 2. Prepare features and labels
X, y, chord_to_idx, idx_to_chord = prepare_training_data(dataset)

# 3. Normalize and split
X_train, X_test, y_train, y_test, scaler = normalize_and_split(X, y)

# 4. Reshape for CNN
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# 5. Build model
model = build_cnn_model(input_shape=(X_train.shape[1], 1), num_classes=len(chord_to_idx))
model.summary()

# 6. Train
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

# 7. Evaluate
loss, acc = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {acc:.4f}")
