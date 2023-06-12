import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np


def run():

    st.subheader('Crack Prediction')
    # st.write('**Model Limitations**')
    st.write('Note : You can only predict : `Crack` or `Not Crack`')
    st.write('')
    model = load_model('improved_model.h5')

    uploaded_file = st.file_uploader(
        'Upload an image', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        # Resize the image to match the input size of the model
        resized_image = image.resize((120, 120))
        # Normalize the pixel values
        normalized_image = np.array(resized_image) / 255.0
        # Add an extra dimension as the model expects a batch of images
        input_image = np.expand_dims(normalized_image, axis=0)

        # Perform image classification
        prediction = model.predict(input_image)
        predicted_class = []

        for pred in prediction:
            if pred >= 0.5:
                predicted_class.append(1)
            else:
                predicted_class.append(0)

        if predicted_class[0] == 1:
            st.write('Prediction: ' + '`Crack`')
        else:
            st.write('Prediction: ' + '`Not Crack`')


if __name__ == '__main__':
    run()
