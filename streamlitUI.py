
import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import ast
from fpdf import FPDF
import io
from PIL import Image

encoders = pd.read_csv("encoders.csv")
encoders_dict = {}
for column in encoders.columns:
    encoders_dict[column] = dict(encoders[column].dropna().to_dict())

image_view = []
left_or_right_breast=[]
calc_type=[]
calc_distribution=[]
for feature, data in encoders_dict.items():
    if 1 in data:
        feature_dict_str = data[1].replace('nan', 'None')
        feature_dict = eval(feature_dict_str)
        feature_list = list(feature_dict.keys())
        globals()[feature] = feature_list

special_labels = ['assessment', 'breast_density']

mammo_model = load_model("models/mammo_model.h5")
pathology_model=load_model("models/pathology_model.h5")
multi_label_model = load_model("models/multi_label_classification_model.h5")


def preprocess_image_patho(uploaded_image):
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8),  cv2.IMREAD_COLOR)
    img = cv2.resize(img,(50, 50))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img


def preprocess_image_mammo(uploaded_image):
    img = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), -1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to 3 channels (RGB)
    img = cv2.resize(img, (224, 224))  # Resize to the correct shape
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 224, 224, 3)
    return img

def predict_mammo(image):
    binary_prediction = mammo_model.predict(image)
    if binary_prediction > 0.1:
        return 1
    else:
        return 0

def predict_pathology(image):
    prob = pathology_model.predict(image)
    prediction =0 if prob<=0.5 else 1
    return prediction

def mri_visualize (uploaded_image):
  image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8),  cv2.IMREAD_COLOR)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  normalized = cv2.normalize(gray, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

  tumor_threshold = 0.5
  normal_threshold = 0.2

  heatmap = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)

  heatmap[(normalized <= normal_threshold)] = [255,0,255 ]
  heatmap[(normalized > normal_threshold) & (normalized < tumor_threshold)] = [0, 255, 255]
  heatmap[(normalized >= tumor_threshold)] = [255, 0, 0]
  tint = cv2.addWeighted(image,0.5,np.full_like(image, (0, 255, 0)),0.5,0)

  overlay = cv2.addWeighted(tint, 0.7, heatmap, 0.3, 0)

  return image,heatmap,overlay

def generate_report(image):
    multi_label_prediction = multi_label_model.predict(image)

    report = f"""
                                                            Full Report

Findings:
The mammographic examination reveals that the image was taken using the {image_view[np.argmax(multi_label_prediction[0])]} view.
The evaluation of the breast indicates that the {left_or_right_breast[np.argmax(multi_label_prediction[1])]} side was examined.
{("No calcifications were observed, and therefore, no distribution pattern is noted." if calc_type[np.argmax(multi_label_prediction[2])] == None
else f"Regarding calcifications, the type identified is {calc_type[np.argmax(multi_label_prediction[2])]}. The calcification distribution is classified as {calc_distribution[np.argmax(multi_label_prediction[3])]}.")}
The breast density assessment resulted in a classification of {np.argmax(multi_label_prediction[4])}, which may have implications for screening sensitivity.
Cancer Assessment:
The analysis suggests a cancer assessment score of {np.argmax(multi_label_prediction[5])}. Further evaluation and correlation with clinical findings, as well as additional imaging if necessary, are recommended.

Conclusion:
These findings should be interpreted in conjunction with the patient's clinical history and prior imaging studies. Consultation with a radiologist or oncologist is advised for further assessment if needed.

  """
    return report

def generate_pdf(report_text, images):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, report_text)
    for img_title, img in images.items():
        pdf.add_page()
        pdf.set_font("Arial", style='B', size=14)
        pdf.cell(200, 10, img_title, ln=True, align='C')

        img_path = f"temp_{img_title}.jpg"
        img.save(img_path)
        pdf.image(img_path, x=10, y=30, w=180)

    pdf_output = pdf.output(dest='S').encode('latin1')

    return io.BytesIO(pdf_output)

def predict_other_features(image):
    # Predict using the multi-label model
    multi_label_prediction = multi_label_model.predict(image)

    predictions = []
    for idx, prediction in enumerate(multi_label_prediction):
        if idx == 5:
            predictions.append("Cancer Assessment: " + str(np.argmax(prediction)))
        elif idx == 4:
            predictions.append("Breast Density Number: " + str(np.argmax(prediction)))
        else:
            max_index = np.argmax(prediction)
            if idx == 0:
                predictions.append(f"Mammography Image View: {image_view[max_index]}")
            elif idx == 1:

                predictions.append(f"Side of The Breast: {left_or_right_breast[max_index]}")
            elif idx == 2:
                predictions.append(f"Calcification Type: {calc_type[max_index]}")
            elif idx == 3:
                if predictions[2].split(':')[1]== ' None':
                   predictions.append(f"Calcification Distribution: None")
                else:
                   predictions.append(f"Calcification Distribution: {calc_distribution[max_index]}")

    return predictions


st.title("Comprehensive Tumor Detection🎗️ 📊🩻🔬")
st.write("Please Upload a breast tumor image for prediction📈")

uploaded_mammo = st.file_uploader("Please Upload Mammography Image...", type=["jpg", "png"])

if uploaded_mammo is not None:
  mammo_image = preprocess_image_mammo(uploaded_mammo)

  st.subheader("Please Wait For The Results Of The Mammography📊")
  Mammo_result = predict_mammo(mammo_image)
  if Mammo_result==0:
    st.write("Benign ✅ No Cancer Detected🎉")

  if Mammo_result == 1:
    st.subheader("Warning ⚠️ Possible Cancerous Mass 🤱")
    st.image(uploaded_mammo, caption="Uploaded Mammography", use_container_width=True)
    st.subheader("Please make a Pathology test and upload the result🔬")
    uploaded_patho = st.file_uploader("Choose Pathology Image...", type=["jpg", "png"])
    if uploaded_patho is not None:
      patho_image = preprocess_image_patho(uploaded_patho)
      st.subheader("Please Wait For The Results Of The Pathology 📊")

      Patho_result = predict_pathology(patho_image)

      if Patho_result==0:
        st.write("Benign ✅ No Cancer Detected🎉")
      if Patho_result==1:
        st.subheader("Cancer Detected 💔 don't worry, we're here for you 💓")
        st.image(uploaded_patho, caption="Uploaded Pathology", use_container_width=True)

        st.subheader("Please Upload MRI For review, We're with you every step of the way ☺️")
        uploaded_mri = st.file_uploader("Choose MRI Image...", type=["jpg", "png"])
        if uploaded_mri is not None:
          image,heatmap,overlay = mri_visualize(uploaded_mri)
          st.subheader("Here's The Visualization Of The Cancer in The MRI📊")
          col1, col2, col3 = st.columns(3)
          with col1:
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Original MRI", use_container_width=True)
          with col2:
            st.image(heatmap, caption="Heatmap", use_container_width=True)
          with col3:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Overlay", use_container_width=True)

          st.subheader("Please Wait while Generating The Full Report:")
          report_text = generate_report(mammo_image)

          st.subheader("Let's Start your healing journy queen👩🏻‍🦰👑")
          images = {
                        "Mammography Image": Image.open(uploaded_mammo),
                        "Pathology Image": Image.open(uploaded_patho),
                        "Original MRI": Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                        "Heatmap": Image.fromarray(heatmap),
                        "Overlay": Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)),
                    }
          pdf_file = generate_pdf(report_text, images)
          st.download_button(label="Download Full Report", data=pdf_file, file_name="Breast_Cancer_Report.pdf", mime="application/pdf")



    st.markdown("""
        <style>
            .css-1v0mbdj {background-color: #f7f7f7; padding: 10px; border-radius: 10px;}
            h1 {color: #6a1b9a;}
            h2 {color: #d32f2f;}
            .stButton button {background-color: #6a1b9a; color: white; border-radius: 5px;}
        </style>
    """, unsafe_allow_html=True)
