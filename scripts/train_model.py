import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


# =========================
# 1. LOAD DATA
# =========================
print("Loading data...")

X_train = np.load("X_train_California.npy")
Y_train = np.load("Y_train_California.npy")

X_test = np.load("X_test_California.npy")
Y_test = np.load("Y_test_California.npy")

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)


# =========================
# 2. ENCODE LABELS
# =========================
print("Encoding labels...")

le = LabelEncoder()
Y_train = le.fit_transform(Y_train)
Y_test = le.transform(Y_test)

print("Classes:", len(le.classes_))


# =========================
# 3. MODEL
# =========================
print("Training model...")

model = RandomForestClassifier(
    n_estimators=150,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, Y_train)


# =========================
# 4. PREDICTION
# =========================
print("Predicting...")

Y_pred = model.predict(X_test)


# =========================
# 5. ACCURACY
# =========================
acc = accuracy_score(Y_test, Y_pred)
print("Accuracy:", acc)


# =========================
# 6. CONFUSION MATRIX
# =========================
print("Plotting confusion matrix...")

cm = confusion_matrix(Y_test, Y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# =========================
# 7. FEATURE IMPORTANCE
# =========================
print("Plotting feature importance...")

features = ["B02 (Blue)", "B03 (Green)", "B04 (Red)", "B08 (NIR)"]
importances = model.feature_importances_

plt.figure()
plt.bar(features, importances)
plt.title("Feature Importance (Sentinel-2 Bands)")
plt.xlabel("Bands")
plt.ylabel("Importance")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# =========================
# 8. CLASS DISTRIBUTION
# =========================
print("Plotting class distribution...")

unique, counts = np.unique(Y_train, return_counts=True)

plt.figure()
plt.bar(unique, counts)
plt.title("Distribution des classes (Train)")
plt.xlabel("Classe")
plt.ylabel("Nombre de pixels")
plt.tight_layout()
plt.show()


# =========================
# 9. SAMPLE PREDICTIONS
# =========================
print("Visualizing predictions...")

n = 50
plt.figure(figsize=(10, 4))

plt.plot(Y_test[:n], label="True", marker='o')
plt.plot(Y_pred[:n], label="Pred", marker='x')

plt.title("Comparaison Y_true vs Y_pred")
plt.legend()
plt.tight_layout()
plt.show()


print(" DONE")