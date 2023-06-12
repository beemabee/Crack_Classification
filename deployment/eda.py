import streamlit as st
import os
import glob
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title='Crack Predictor',
    layout='wide',
    initial_sidebar_state='expanded'
)


def run():
    # Plot title
    st.title('Crack Prediction')

    # Plot sub-header
    st.subheader('Explore the Filepath and Label from Dataset')

    markdown_text = '''
    ## Background
    The construction industry plays a crucial role in infrastructure development, and 
    ensuring the structural integrity of buildings and infrastructure is of utmost 
    importance. One common issue faced in the construction field is the detection and 
    prediction of cracks in structures. Timely detection of cracks can help prevent 
    accidents, reduce maintenance costs, and ensure the safety of occupants.

    In this context, the development of a crack prediction model using Convolutional 
    Neural Networks (CNN) holds significant value. By leveraging advanced deep learning 
    techniques, such a model can effectively analyze images of structures and accurately 
    predict the presence of cracks. This enables early identification of potential
    structural issues, allowing for timely repairs and maintenance. The model can assist 
    construction professionals in assessing the condition of buildings, bridges, and 
    other infrastructure, aiding in decision-making processes, prioritizing repairs, 
    and optimizing resource allocation. Ultimately, the application of this modeling 
    approach can contribute to safer, more durable construction practices and enhance 
    the overall quality and longevity of infrastructure in the construction industry.
    
    ## Objective
    Preparing the dataset with preprocessing, exploring the dataset, analyzing, 
    and implementing Convolutional Neural Network (CNN) to predict the presence 
    of cracks in an area, and then improving the model.
    
    ## About Dataset
    The dataset used in this project consists of images that have been categorized as 
    positive (defective or construction issue) and negative (non-defective) images. 
    This dataset can be utilized to train a model for automatic classification and 
    prediction of construction images.

    By training the model on this dataset, it can learn the distinguishing features and 
    patterns that differentiate defective and non-defective images. This enables the 
    model to classify new construction images and predict whether they contain any 
    issues or defects.
    
    '''
    st.markdown(markdown_text)
    st.markdown('---')

    # define main_path to know the folder
    # turn the directory path into path object
    positive_dir = Path('Positive')
    negative_dir = Path('Negative')

    # create dataframe
    def generate_df(image_dir, label):
        filepaths = pd.Series(list(image_dir.glob(r'*.jpg')),
                              name='Filepath').astype(str)
        labels = pd.Series(label, name='Label', index=filepaths.index)
        df = pd.concat([filepaths, labels], axis=1)
        return df

    # Check Positive-Negative DataFrame
    positive_df = generate_df(positive_dir, 'POSITIVE')
    negative_df = generate_df(negative_dir, 'NEGATIVE')

    # merge dataframe
    data_df = pd.concat([positive_df, negative_df], axis=0).sample(
        frac=1, random_state=42).reset_index(drop=True)

    data_df['Label'] = data_df['Label'].replace(
        {'NEGATIVE': 'Not Crack', 'POSITIVE': 'Crack'})

    st.dataframe(data_df)

    # Distribusi dataset
    fig, ax = plt.subplots(ncols=1, figsize=(12, 6))
    sns.countplot(x='Label', data=data_df, palette='viridis')
    ax.set_title('Crack-Not Crack Image Distribution', fontsize=18)
    st.pyplot(fig)

    st.write('This dataset contains a total of 40,000 images, with 20,000 images depicting areas with cracks and another 20,000 images representing normal areas.')

    st.markdown('---')

    st.write('## Cracked Surface')
    st.write('Below are some examples of some areas that are `Cracked`')

    def visualize_samples_by_label(df, label, num_samples=20):
        samples = df[df['Label'] ==
                     label]['Filepath'].iloc[:num_samples].tolist()
        num_cols = min(num_samples, 5)
        num_rows = (num_samples - 1) // num_cols + 1
        fig, axes = plt.subplots(
            nrows=num_rows, ncols=num_cols, figsize=(10, 2 * num_rows))
        count = 0
        for i in range(num_rows):
            for j in range(num_cols):
                if count < len(samples):
                    sample = samples[count]
                    img = Image.open(sample)
                    ax = axes[i, j]
                    ax.imshow(img)
                    ax.axis('off')
                    count += 1
        plt.tight_layout()
        st.pyplot(fig)

    visualize_samples_by_label(data_df, 'Crack', num_samples=20)

    st.markdown('---')

    st.write('## Normal Surface')
    st.write(
        'Below are some examples of some areas that are `normal` or `not cracked`')

    visualize_samples_by_label(data_df, 'Not Crack', num_samples=20)
    markdown_text = '''
    The data sampling above indicates that in the dataset, both the normal and crack 
    images predominantly consist of a combination of brown, white, and gray colors. 
    The crack areas are located in the middle portion of the images, dividing them into 
    two parts, either vertically or horizontally.
    '''
    st.markdown('---')
    st.write(markdown_text)

    st.markdown('---')
    st.write('## Image Transformation')
    st.markdown('---')

    st.write('### 1. HUE')
    st.markdown('---')
    # tambah gambar hue
    image = Image.open('hue.png')
    st.image(image)
    markdown_text = '''
    From the dataset, it can be observed that for the "Not Crack" label, the majority 
    of the generated colors are blue, but there are also instances where the colors are 
    predominantly purple. This may be correlated with the intensity of light hitting the 
    normal surface area, resulting in colors other than blue. 
    
    As for the "Crack" label, 
    most of the samples exhibit a blue color for the areas without cracks. However, the 
    cracked areas are characterized by different colors and patterns, such as red or 
    green.
    '''
    st.write(markdown_text)
    st.markdown('---')

    st.write('### 2. LUV Color Image')
    st.markdown('---')

    # tambah gambar LUV
    image = Image.open('LUV.png')
    st.image(image)
    markdown_text = '''
    rom the dataset, it can be observed that the majority of images, both in the normal 
    and crack categories, exhibit a reddish color. However, in the cracked areas, there 
    is a distinct black color indicating the presence of cracks on the surface.
    '''
    st.write(markdown_text)
    st.markdown('---')

    st.write('### 3. Grayscale')
    st.markdown('---')

    # tambah gambar grayscale
    image = Image.open('grayscale.png')
    st.image(image)
    markdown_text = '''
    In the grayscale results with the original RGB dataset, there doesn't appear to be a 
    significant difference. This is likely due to the background colors in the dataset 
    being a combination of white, brown, and gray. In the areas labeled as cracks, there 
    is a presence of dark or black color indicating the presence of cracks in those areas.
    '''
    st.write(markdown_text)
    st.markdown('---')


if __name__ == '__main__':
    run()
