import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def run_model():
    print("🚀 Loading dataset...")

    # Load dataset
    dataset = pd.read_csv("cancer.csv")
    print(
        f"✅ Dataset loaded successfully with {len(dataset)} rows and {len(dataset.columns)} columns.\n"
    )

    # Split features and labels
    x = dataset.drop(columns=["diagnosis(1=m, 0=b)"])
    y = dataset["diagnosis(1=m, 0=b)"]

    print("📊 Splitting data into training and testing sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    print(
        f"✅ Data split complete: {x_train.shape[0]} training samples, {x_test.shape[0]} test samples.\n"
    )

    # Build the model
    print("🧠 Building the neural network...")
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                256, input_shape=(x_train.shape[1],), activation="sigmoid"
            ),
            tf.keras.layers.Dense(256, activation="sigmoid"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    print("✅ Model compiled successfully.\n")

    # Train the model
    print("🏋️ Training the model (this may take a moment)...")
    history = model.fit(x_train, y_train, epochs=100, verbose=1)
    print("✅ Training complete!\n")

    # Evaluate the model
    print("🧾 Evaluating model performance on test data...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(
        f"✅ Model evaluation complete.\n📈 Accuracy: {accuracy:.4f} | Loss: {loss:.4f}\n"
    )

    # 🔍 Generate predictions
    print("🔍 Generating predictions and confusion matrix...")
    y_pred = model.predict(x_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    # 🧾 Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    print("\n🧾 Confusion Matrix:")
    print(cm)

    # 📊 Print classification report
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred_classes))

    # 📈 Visualize confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Benign (0)", "Predicted Malignant (1)"],
        yticklabels=["Actual Benign (0)", "Actual Malignant (1)"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()

    # Save confusion matrix image
    plt.savefig("images/confusion_matrix.png")
    plt.show()

    # 💾 Save trained model
    model.save("cancer_model.keras")
    print("\n💾 Model saved successfully as 'cancer_model.keras'.")
    print(
        "🎉 All done! Model trained, evaluated, and confusion matrix generated successfully."
    )


def main():
    print("Hello from cancer!")
    run_model()


if __name__ == "__main__":
    main()
