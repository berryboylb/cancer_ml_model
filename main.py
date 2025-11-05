import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf


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

    print("🎉 All done! Model trained and evaluated successfully.")


def main():
    print("Hello from cancer!")
    run_model()


if __name__ == "__main__":
    main()
