import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50

# Define class labels
class_labels = ["Actinic keratoses (AKIEC)",
                "Melanocytic nevi (NV)",
                "Basal cell carcinoma (BCC)",
                "Benign keratosis-like lesions (BKL)",
                "Dermatofibroma (DF)",
                "Melanoma (MEL)",
                "Vascular lesions (VASC)"]

# Load the trained model
model_path = "skinmodel50.pt"  # Path to your trained .pt file
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(model_path, map_location=device)

# Instantiate ResNet50 model
model = resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(class_labels))

# Load the model state dictionary
model.load_state_dict(checkpoint)

# Set the model to evaluation mode
model.eval()

# Define preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to make predictions
def predict(image):
    # Preprocess the image
    image_tensor = preprocess(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_class = class_labels[predicted.item()]
    
    return predicted_class

# Streamlit app
st.title("Skin Cancer Classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction when button is clicked
    if st.button("Classify"):
        # Make prediction
        prediction = predict(image)
        st.success(f"Lesion Class: {prediction}")
