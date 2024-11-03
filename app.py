import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import io

# Load the model
# Define class names and detailed descriptions with treatment suggestions
classes = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Tooth Discoloration', 'Ulcers', 'Healthy Teeth']
disease_info = {
    'Calculus': {
        "description": "Calculus, also known as tartar, is hardened dental plaque that has been mineralized by calcium phosphate deposits.",
        "causes": "It is caused by the accumulation of plaque due to irregular brushing and flossing.",
        "treatment": "Professional dental cleaning is required to remove calculus deposits."
    },
    'Caries': {
        "description": "Dental caries, or cavities, are permanent damage to the enamel that develop into tiny holes in the teeth.",
        "causes": "Caries are caused by acid-producing bacteria in the mouth, poor oral hygiene, and high sugar intake.",
        "treatment": "Regular brushing and a visit to the dentist for potential fillings or fluoride treatments."
    },
    'Gingivitis': {
        "description": "Gingivitis is inflammation of the gums, often a mild form of gum disease.",
        "causes": "It is commonly caused by poor oral hygiene, which allows plaque to build up on the teeth.",
        "treatment": "Improving oral hygiene and using antiseptic mouthwash can help; a professional cleaning may also be needed."
    },
    'Hypodontia': {
        "description": "Hypodontia is a condition where one or more teeth are missing due to developmental issues.",
        "causes": "It can be caused by genetic factors or developmental disorders.",
        "treatment": "Consult a dentist for possible orthodontic treatment, bridges, or implants."
    },
    'Tooth Discoloration': {
        "description": "Tooth discoloration is the staining or yellowing of teeth, which can be extrinsic (surface stains) or intrinsic (within the tooth).",
        "causes": "Common causes include consumption of staining foods, smoking, aging, and poor dental hygiene.",
        "treatment": "Consider whitening options or a dental checkup to rule out underlying issues."
    },
    'Ulcers': {
        "description": "Oral ulcers are painful sores that occur inside the mouth, often on the inner cheeks or lips.",
        "causes": "They can be caused by stress, injury, or certain foods, and may be associated with other health conditions.",
        "treatment": "Rinse with salt water, use prescribed mouth gels, and consult a dentist if they persist."
    },
    'Healthy Teeth': {
        "description": "No signs of dental issues. Your teeth appear healthy.",
        "causes": "Maintained through good oral hygiene and regular dental checkups.",
        "treatment": "Continue with proper brushing, flossing, and regular visits to the dentist."
    }
}

# Streamlit app
def main():
    st.title("Oral Vision: A Oral Disease Detector")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file).convert('RGB')
        
        # Show the image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load the model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_name = 'efficientvit_b0_oral_disease_classifier.pth'  # Default model name
        model = load_model(model_name, len(classes) - 1, device)  # Model without "Healthy Teeth" class

        # Preprocess the image
        processed_image = preprocess_image(image).to(device)

        # Predict
        with torch.no_grad():
            outputs = model(processed_image)
            probabilities = torch.softmax(outputs, dim=1)
            max_prob, predicted = torch.max(probabilities, 1)
            prediction = predicted.item()

        # Check confidence threshold for "Healthy Teeth" classification
        confidence_threshold = 0.7  # You can adjust this threshold based on your dataset
        if max_prob < confidence_threshold:
            disease = "Healthy Teeth"
        else:
            disease = classes[prediction]

        # Display the prediction and detailed information
        st.write(f"Prediction: {disease}")
        st.write("Description:", disease_info[disease]["description"])
        st.write("Causes:", disease_info[disease]["causes"])
        st.write("Suggested Treatment:", disease_info[disease]["treatment"])

if __name__ == "__main__":
    main()
