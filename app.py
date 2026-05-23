from fastapi import FastAPI, File, UploadFile
from PIL import Image
import numpy as np
import tensorflow as tf

# Create FastAPI app
app = FastAPI()

# Load trained model
model = tf.keras.models.load_model(
    "image_classifier.h5"
)

# Home route
@app.get("/")
def home():

    return {
        "message": "Image Classification API Running"
    }

# Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Open uploaded image
    image = Image.open(file.file).convert("RGB")

    # Resize image
    image = image.resize((128,128))

    # Convert image to numpy array
    image = np.array(image)

    # Normalize image
    image = image / 255.0

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)[0][0]

    # Convert prediction into class
    if prediction > 0.5:
        result = "Dog"
    else:
        result = "Cat"

    # Return response
    return {
        "prediction": result,
        "confidence": float(prediction)
    }