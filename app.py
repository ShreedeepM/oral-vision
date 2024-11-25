import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import timm
import io

# Load the model
def load_model(model_name, num_classes, device):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    model_path = 'efficientvit_b0.pth'
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

# Define class names and detailed descriptions with treatment suggestions
classes = ['Calculus', 'Caries', 'Gingivitis', 'Hypodontia', 'Tooth Discoloration', 'Ulcers']
disease_info = {
    'Calculus': {
        "description": "Calculus, also known as tartar, is a hardened form of dental plaque that builds up on teeth over time. It provides a surface for additional plaque to form, leading to more buildup and potential gum disease.",
        "causes": "Calculus forms when plaque is not removed effectively, and the minerals in saliva harden it onto the teeth. Inadequate brushing and flossing, along with high-calcium saliva, increase the risk.",
        "treatment": "Professional cleaning by a dentist or dental hygienist is necessary to remove calculus deposits. Regular brushing and flossing help prevent its formation."
    },
    'Caries': {
        "description": "Dental caries, or cavities, are areas of tooth decay that create tiny openings or holes in the enamel, leading to pain and potential tooth loss if untreated.",
        "causes": "Caries are caused by bacteria in the mouth that produce acids from sugars in food, which then erode tooth enamel. Poor oral hygiene, frequent snacking, and high sugar intake increase the risk.",
        "treatment": "Treatment includes fluoride treatments for early-stage caries, fillings for moderate decay, and crowns or root canals for advanced decay. Good oral hygiene and reducing sugar intake can help prevent caries."
    },
    'Gingivitis': {
        "description": "Gingivitis is a mild form of gum disease characterized by red, swollen, and bleeding gums. It is reversible with proper treatment but can lead to more severe gum disease if left untreated.",
        "causes": "Gingivitis is commonly caused by plaque buildup along the gum line due to inadequate brushing and flossing. Hormonal changes, smoking, and certain medications can also contribute.",
        "treatment": "Improving oral hygiene and professional cleaning can usually reverse gingivitis. Antiseptic mouthwashes and regular dental checkups are also recommended for management."
    },
    'Hypodontia': {
        "description": "Hypodontia is a dental condition where one or more teeth fail to develop. It is a genetic condition that may affect primary or permanent teeth, potentially leading to spacing issues and misalignment.",
        "causes": "Hypodontia is caused by genetic factors and may be associated with certain syndromes. Developmental factors during early tooth formation can also play a role.",
        "treatment": "Treatment options include orthodontics to close gaps, dental bridges, or implants to replace missing teeth. Consultation with a dentist is recommended for individualized care."
    },
    'Tooth Discoloration': {
        "description": "Tooth discoloration refers to the staining or darkening of teeth, which can occur on the surface (extrinsic) or within the tooth (intrinsic). It affects aesthetics and may signal other health issues.",
        "causes": "Extrinsic discoloration is caused by foods, drinks, and smoking. Intrinsic discoloration can result from trauma, aging, certain medications, and excessive fluoride exposure.",
        "treatment": "Treatment includes professional cleaning for extrinsic stains and whitening options for intrinsic stains. Maintaining good oral hygiene and avoiding staining substances can help prevent discoloration."
    },
    'Ulcers': {
        "description": "Oral ulcers, commonly known as canker sores, are painful lesions that appear on the soft tissues inside the mouth. They can make eating and speaking uncomfortable.",
        "causes": "Causes include minor injuries, stress, certain acidic or spicy foods, and underlying health conditions. Frequent or recurrent ulcers may require further evaluation.",
        "treatment": "Rinsing with salt water, avoiding spicy or acidic foods, and using prescribed mouth gels can alleviate discomfort. Persistent ulcers should be examined by a dentist or doctor."
    }
}

# Streamlit app
def main():
    st.title("Oral Vision: An Oral Disease Detector")
    
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

        # Get the predicted disease
        disease = classes[prediction]

        # Display the prediction and detailed information
        st.write(f"**Prediction:** {disease}")
        st.write(f"**Description:** {disease_info[disease]['description']}")
        st.write(f"**Causes:** {disease_info[disease]['causes']}")
        st.write(f"**Suggested Treatment:** {disease_info[disease]['treatment']}")

if __name__ == "__main__":
    main()
