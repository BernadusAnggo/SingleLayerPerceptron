#------------------------------------------------------------
# Tugas Single Layer Perceptron
# Bernadus Anggo Seno Aji
# 25/571654/SPA/01170
# DOKTOR ILMU KOMPUTER
#------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1) Inisialisasi bobot & bias
# -----------------------------
def init_params(input_dim, init_w=0.5, init_b=0.5):
    weights = np.full(input_dim, init_w)
    bias = init_b
    return weights, bias

# -----------------------------------------
# 2) Dot product bobot & bias
# -----------------------------------------
def dot_product(x, weights, bias):
    return np.sum(x * weights) + bias

# -----------------------------
# 3) Fungsi aktivasi (sigmoid)
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -------------------
# 4) Hitung Error MSE
# -------------------
def squared_error(y_true, y_pred):
    return (y_true - y_pred) ** 2

# --------------------------------------------------------
# 5) Gradient descent dan update bobot dan bias
# --------------------------------------------------------
def update_params(weights, bias, x_i, y_true, y_pred, learning_rate):
    # gradient descent
    gradien = 2 * (y_pred - y_true) * (1 - y_pred) * y_pred
    grad_w = gradien * x_i
    grad_b = gradien * 1
    # update bobot dan bias
    weights = weights - learning_rate * grad_w
    bias = bias - learning_rate * grad_b
    return weights, bias

if __name__ == "__main__":
    # Load data iris
    df = pd.read_csv("Iris.csv", header=None,
                     names=["sepal_length", "sepal_width", "petal_length", "petal_width", "species"])

    # Filter hanya dua kelas: setosa dan versicolor
    df = df[df["species"].isin(["Iris-setosa", "Iris-versicolor"])]

    # Replace label: setosa=0, versicolor=1
    df["label"] = df["species"].map({"Iris-setosa": 0, "Iris-versicolor": 1})

    # Ambil fitur, label
    X = df.iloc[:, 0:4].values
    y = df["label"].values

    # -----------------------------
    # Split per kelas (1–50 kelas 0, 51–100 kelas 1; ambil 80% masing-masing)
    # -----------------------------
    X_class0, y_class0 = X[:50], y[:50]
    train_size0 = int(0.8 * len(X_class0))
    X_train0, X_val0 = X_class0[:train_size0], X_class0[train_size0:]
    y_train0, y_val0 = y_class0[:train_size0], y_class0[train_size0:]

    X_class1, y_class1 = X[50:100], y[50:100]
    train_size1 = int(0.8 * len(X_class1))
    X_train1, X_val1 = X_class1[:train_size1], X_class1[train_size1:]
    y_train1, y_val1 = y_class1[:train_size1], y_class1[train_size1:]

    # Gabungkan train & val
    X_train = np.vstack((X_train0, X_train1))
    y_train = np.hstack((y_train0, y_train1))
    X_validasi = np.vstack((X_val0, X_val1))
    y_validasi = np.hstack((y_val0, y_val1))

    # Inisialisasi bobot dan bias 0.5, learning rate 0.1
    weights, bias = init_params(input_dim=X.shape[1], init_w=0.5, init_b=0.5)
    learning_rate = 0.1
    epochs = 5

    # Array untuk loss dan akurasi
    data_loss_train = []
    accuracy_train = []
    data_loss_validasi = []
    accuracy_validasi = []

    # Perulangan sebanyak epoch
    for epoch in range(epochs):
        total_loss_train = 0
        total_loss_validasi = 0
        acc_train = 0
        acc_validasi = 0

        # --------------------------------
        # Training
        # --------------------------------
        for i in range(len(X_train)):
            x_i = X_train[i]
            y_true = y_train[i]

            # dot product bobot dan bias data training
            z = dot_product(x_i, weights, bias)

            # ubah z ke dalam fungsi aktivasi (sigmoid)
            y_pred = sigmoid(z)

            # hitung Error
            total_loss_train += squared_error(y_true, y_pred)

            # kategori prediksi
            y_pred_cat = 1 if y_pred >= 0.5 else 0

            # akurasi training
            if y_pred_cat == y_true:
                acc_train += 1

            # Gradient descent dan Update bobot & bias
            weights, bias = update_params(weights, bias, x_i, y_true, y_pred, learning_rate)

        # simpan loss & akurasi training per-epoch
        mean_loss_train = total_loss_train / len(X_train)
        data_loss_train.append(float(mean_loss_train))
        accuracy_train.append(acc_train / len(X_train) * 100)

        # --------------------------------
        # Validasi
        # --------------------------------
        for i in range(len(X_validasi)):
            x_val = X_validasi[i]
            y_val_true = y_validasi[i]

            z_val = dot_product(x_val, weights, bias)
            y_val_pred = sigmoid(z_val)

            total_loss_validasi += squared_error(y_val_true, y_val_pred)

            y_val_cat = 1 if y_val_pred >= 0.5 else 0
            if y_val_cat == y_val_true:
                acc_validasi += 1

        mean_loss_val = total_loss_validasi / len(X_validasi)
        data_loss_validasi.append(float(mean_loss_val))
        accuracy_validasi.append(acc_validasi / len(X_validasi) * 100)

    # visualisasi loss dan akurasi setiap epoch
    plt.figure(figsize=(8, 6))
    epochs_axis = range(1, epochs + 1)

    # Subplot 1: Loss training
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(epochs_axis, data_loss_train, marker='o')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean Squared Error")
    ax1.set_title("Training Loss")
    for x, y in zip(epochs_axis, data_loss_train):
        ax1.annotate(f"{float(y):.4f}", (x, y),
                    textcoords="offset points", xytext=(0, 5),
                    ha='center', fontsize=8)

    # Subplot 2: Akurasi training
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(epochs_axis, accuracy_train, marker='o')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Training Accuracy")
    for x, y in zip(epochs_axis, accuracy_train):
        ax2.annotate(f"{float(y):.2f}", (x, y),
                    textcoords="offset points", xytext=(0, 5),
                    ha='center', fontsize=8)

    # Subplot 3: Loss validasi
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(epochs_axis, data_loss_validasi, marker='o')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Mean Squared Error")
    ax3.set_title("Validasi Loss")
    for x, y in zip(epochs_axis, data_loss_validasi):
        ax3.annotate(f"{float(y):.4f}", (x, y),
                    textcoords="offset points", xytext=(0, 5),
                    ha='center', fontsize=8)

    # Subplot 4: Akurasi validasi
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(epochs_axis, accuracy_validasi, marker='o')
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Accuracy (%)")
    ax4.set_title("Validasi Accuracy")
    for x, y in zip(epochs_axis, accuracy_validasi):
        ax4.annotate(f"{float(y):.2f}", (x, y),
                    textcoords="offset points", xytext=(0, 5),
                    ha='center', fontsize=8)

    plt.tight_layout()
    plt.show()