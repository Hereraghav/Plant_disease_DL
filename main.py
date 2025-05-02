import os
import json
import torch
import streamlit as st
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# 📂 Working Directory
working_dir = os.path.dirname(os.path.abspath(__file__))

# 📑 Load class indices
class_indices_path = os.path.join(working_dir, 'class_indices.json')
with open(class_indices_path, 'r') as f:
    class_indices = json.load(f)

# 🔁 Inverse class indices
idx_to_class = {int(v): k for v, k in class_indices.items()}

# 🧠 Load model (same as training)
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    
    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2, 2),
    
    nn.Flatten(),
    nn.Linear(64 * 56 * 56, 256),
    nn.ReLU(),
    nn.Linear(256, len(class_indices))
)

model_path = os.path.join(working_dir, 'plant_disease_model.pth')
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# 📸 Image preprocessing
def load_and_preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    image = image.convert('RGB')
    img_tensor = transform(image)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

# 🔎 Prediction function
def predict_image_class(model, image, idx_to_class):
    img_tensor = load_and_preprocess_image(image)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_idx = predicted.item()
    return idx_to_class[predicted_idx]

# 🌟 Streamlit App
st.set_page_config(page_title="Potato Leaf Disease Detection 🌿", page_icon="🌱", layout="centered")

st.title('🌿 Potato Leaf Disease Detection')
st.markdown("""
Welcome to the Potato Leaf Disease Detection App!  
Upload a leaf image, and we'll detect if it's healthy or infected. 🌱🩺
""")

uploaded_image = st.file_uploader("📤 Upload a potato leaf image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    
    # ➡️ Resize the uploaded image to smaller size (150x150 like you had earlier)
    resized_image = image.resize((150, 150))
    st.image(resized_image, caption="Uploaded Image", use_container_width=False)

    if st.button('🔍 Classify'):
        prediction = predict_image_class(model, image, idx_to_class)

        st.success(f"### 🩺 Prediction: **{prediction}**")

        st.markdown("---")
        
        if prediction == "Potato___Early_blight":
            st.subheader("🌿 About Early Blight")
            st.write("""
            - **Early Blight** is caused by the fungus *Alternaria solani*.
            - Symptoms: Dark brown spots on leaves, concentric rings.
            - Can lead to defoliation and reduced yield.
            """)
            st.subheader("🛡️ How to Protect:")
            st.write("""
            - Use certified disease-free seeds.
            - Practice crop rotation.
            - Apply appropriate fungicides.
            """)
        
        elif prediction == "Potato___Late_blight":
            st.subheader("🌿 About Late Blight")
            st.write("""
            - **Late Blight** is caused by the oomycete *Phytophthora infestans*.
            - Symptoms: Water-soaked lesions, white moldy growth under leaves.
            - This disease led to the Irish Potato Famine!
            """)
            st.subheader("🛡️ How to Protect:")
            st.write("""
            - Remove infected plants immediately.
            - Use resistant potato varieties.
            - Ensure good airflow in fields.
            """)
        
        elif prediction == "Potato___healthy":
            st.balloons()
            st.subheader("🎉 Your plant looks **healthy**!")
            st.write("""
            - Keep monitoring regularly.
            - Maintain good watering and spacing practices.
            """)

st.markdown("---")
st.caption("Made by Raghav")
