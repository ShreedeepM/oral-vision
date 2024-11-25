import torch
import pickle
import timm
from PIL import Image
from torchvision import transforms
import streamlit as st

# Load model and configuration from pk1 file
with open("model_and_config.pk1", "rb") as f:
    config = pickle.load(f)

# Create model instance
model = timm.create_model("rexnet_150", pretrained=False, num_classes=len(config['classes']))

# Load model state dict with map_location to CPU
model.load_state_dict(config['model_state_dict'], map_location=torch.device('cpu'))
model.eval()  # Set the model to evaluation mode

# Define classes from config
classes = config['classes']

# Define image preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
])

def predict_image(image):
    input_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():  # Disable gradient calculation
        output = model(input_tensor)  # Forward pass
        _, predicted = torch.max(output, 1)  # Get the class with the highest score
        predicted_class = classes[predicted.item()]  # Map index to class name
    return predicted_class

# Streamlit UI
st.title("Oral Disease Prediction")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Open image and convert to RGB
    st.image(image, caption="Uploaded Image", use_column_width=True)  # Display uploaded image
    st.write("Classifying...")  # Indicate classification in progress

    predicted_disease = predict_image(image)  # Make prediction
    st.write(f"The predicted oral disease is: **{predicted_disease}**")  # Display result
