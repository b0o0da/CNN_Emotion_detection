from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

model = tf.keras.models.load_model("best_model.keras")
class_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L').resize((75, 75))  # ← Grayscale
    st.image(img, caption='Uploaded Image', use_container_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=-1)  # ← Add channel dimension (75,75,1)
    img_array = np.expand_dims(img_array, axis=0)   # ← Add batch dimension (1,75,75,1)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.write(f"### Predicted Emotion: {predicted_class}")
    st.write("Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_names[i]}: {prob:.2f}")
