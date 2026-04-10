import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model('dr_model_simple.h5')

labels = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
risk = {
    0: "✅ Low Risk - No action needed",
    1: "⚠️ Low Risk - Monitor regularly",
    2: "⚠️ Medium Risk - Consult Doctor",
    3: "🚨 High Risk - Consult Doctor Immediately",
    4: "🚨 Critical - Urgent Medical Attention!"
}

def generate_gradcam(model, img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer("conv2d_3").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), predictions.numpy()

st.title("🔬 Smart Diabetic Retinopathy Screening")
st.write("Upload a retinal fundus image to detect Diabetic Retinopathy")

uploaded_file = st.file_uploader("Choose a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import gdown
import os

# Download model from Google Drive
model_path = 'dr_model_simple.h5'
if not os.path.exists(model_path):
    with st.spinner("Loading model..."):
        gdown.download(
            "https://drive.google.com/uc?id=1TTXoMA7ITBNUnTrOtARNN1U2Qi4AUSDr",
            model_path, quiet=False
        )

model = tf.keras.models.load_model(model_path)

labels = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
risk = {
    0: "✅ Low Risk - No action needed",
    1: "⚠️ Low Risk - Monitor regularly",
    2: "⚠️ Medium Risk - Consult Doctor",
    3: "🚨 High Risk - Consult Doctor Immediately",
    4: "🚨 Critical - Urgent Medical Attention!"
}

def generate_gradcam(model, img_array):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer("conv2d_3").output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), predictions.numpy()

st.title("🔬 Smart Diabetic Retinopathy Screening")
st.write("Upload a retinal fundus image to detect Diabetic Retinopathy")

uploaded_file = st.file_uploader("Choose a retinal image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    with st.spinner("Analyzing..."):
        img = np.array(image)
        img = cv2.resize(img, (128, 128))
        img_array = np.expand_dims(img / 255.0, axis=0)

        heatmap, predictions = generate_gradcam(model, img_array)
        pred_class = np.argmax(predictions[0])
        confidence = predictions[0][pred_class] * 100

        img_display = np.uint8(img)
        heatmap_resized = cv2.resize(heatmap, (128, 128))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img_display, 0.6, heatmap_colored, 0.4, 0)

    st.subheader("Results")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Diagnosis", labels[pred_class])
        st.metric("Confidence", f"{confidence:.1f}%")
        st.write(f"**Risk Level:** {risk[pred_class]}")
    with col2:
        st.image(superimposed, caption="Grad-CAM Heatmap", width=250)