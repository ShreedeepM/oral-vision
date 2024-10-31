import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import io

# Load the model
def load_model(model_name, num_classes, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model_path = f'best_model/{model_name}_oral_disease_classifier.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Image preprocessing function
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Define class names
classes = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Tooth Discoloration', 'Ulcers']

# Streamlit app
def main():
    st.title("Oral Disease Detector")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Show the image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = 'efficientvit_b0'  # Default model name
        model = load_model(model_name, len(classes), device)

        # Preprocess the image
        processed_image = preprocess_image(image).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(processed_image)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.item()

        # Display the prediction
        st.write(f"Prediction: {classes[prediction]}")

if __name__ == "__main__":
    main()
