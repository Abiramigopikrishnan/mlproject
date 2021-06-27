import streamlit as st
import pandas as pd
import pickle
import base64
import numpy as np

#Creating the UI for the application:

#Side Bar
st.sidebar.title("PD Prediction App")
st.sidebar.write("")
st.sidebar.subheader("Welcome to the Parkinson's Disease Prediction App.")
st.sidebar.write("")
st.sidebar.write("This prediction is done using the Support Vector Machine Method of Machine Learning.")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.text("Created by Abirami Gopikrishnan")
#############################################

#Background
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_png_as_page_bg(jpg_file):
    bin_str = get_base64_of_bin_file(jpg_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return
set_png_as_page_bg('bg_image.jpg')
#############################################

#title
st.title('Parkinsons Disease Prediction')
st.subheader("Enter your details below to analyze and predict your result."	,)
#############################################

#Input Details
name = st.text_input("Enter your Name: ","Type Here...")
MDVP_Fo_Hz = st.number_input("Enter the Average vocal fundamental frequency : ",key = "MDVP_Fo_Hz",format="%.6f")
MDVP_Fhi_Hz = st.number_input("Enter the Maximum vocal fundamental frequency : ", key = "MDVP_Fhi_Hz",format="%.6f")
MDVP_Flo_Hz = st.number_input("Enter the Minimum vocal fundamental frequency : ", key="MDVP_Flo_Hz",format="%.6f")
MDVP_Jitter_percent = st.number_input("Enter the MDVP jitter in percentage : ",key = "MDVP_Jitter_Percent",format="%.6f")
MDVP_Jitter_Abs = st.number_input("Enter the MDVP absolute jitter in ms : ", key = "MDVP_Jitter_Abs",format="%.6f")
MDVP_Rap = st.number_input("Enter the MDVP relative amplitude perturbation : ", key = "MDVP_Rap",format="%.6f")
MDVP_Ppq = st.number_input("Enter the MDVP five-point period perturbation quotient : ", key = "MDVP_Ppq",format="%.6f")
Jitter_DDP = st.number_input("Enter the Average absolute difference of differences between jitter cycles : ", key = "Jitter_DDP",format="%.6f")
MDVP_Shimmer = st.number_input("Enter the MDVP local shimmer : ", key = "MDVP_Shimmer",format="%.6f")
MDVP_Shimmer_dB = st.number_input("Enter the MDVP local shimmer in dB : ", key = "MDVP_Shimmer_dB",format="%.6f")
Shimmer_APQ3 = st.number_input("Enter the Three-point amplitude perturbation quotient : ", key = "Shimmer_APQ3",format="%.6f")
Shimmer_APQ5 = st.number_input("Enter the Five-point amplitude perturbation quotient : ", key = "Shimmer_APQ5",format="%.6f")
MDVP_Apq = st.number_input("Enter the MDVP 11-point amplitude perturbation quotient : ", key = "MDVP_Apq",format="%.6f")
Shimmer_DDA = st.number_input("Enter the Average absolute differences between the amplitudes of consecutive periods : ", key = "Shimmer_DDA",format="%.6f")
Nhr = st.number_input("Enter the	Noise-to-harmonics ratio : ", key = "Nhr",format="%.6f")
Hnr = st.number_input("Enter the	Harmonics-to-noise ratio : ", key = "Hnr",format="%.6f")
Rpde = st.number_input("Enter the Recurrence period density entropy measure : ", key = "Rpde",format="%.6f")
Dfa = st.number_input("Enter the Signal fractal scaling exponent of detrended fluctuation analysis : ", key = "Dfa",format="%.6f")
spread1 = st.number_input("Enter the Two nonlinear measures of fundamental : ", key = "spread1",format="%.6f")
spread2 = st.number_input("Enter the Frequency variation : ", key = "spread2",format="%.6f")
d2 = st.number_input("Enter the Signal fractal scaling exponent : ", key = "d2",format="%.6f")
Ppe = st.number_input("Enter the 	Pitch period entropy : ", key = "Ppe",format="%.6f")
#############################################

#load pickle file
svm_classifier = open('svm_model.pkl','rb')
std = open('std_scalar.pkl','rb')
classifier = pickle.load(svm_classifier)
std_scalar = pickle.load(std)
#############################################
#predict the result
submit = st.button("Analyze")
if submit:
    result=(MDVP_Fo_Hz,MDVP_Fhi_Hz,MDVP_Flo_Hz,MDVP_Jitter_percent,MDVP_Jitter_Abs,MDVP_Rap,MDVP_Ppq,Jitter_DDP,MDVP_Shimmer,MDVP_Shimmer_dB,Shimmer_APQ3,Shimmer_APQ5,MDVP_Apq,Shimmer_DDA,Nhr,Hnr,Rpde,Dfa,spread1,spread2,d2,Ppe)
    result = np.asarray(result)
    input_reshaped = result.reshape(1,-1)

    # standardize the data
    std_input = std_scalar.transform(input_reshaped)

    res=classifier.predict(std_input)

    print(res)
    if res == [0]:
        st.balloons()
        st.success("It is predicted that you may not have Parkinsons Disease")
    else:
        st.error("It is predicted that you may have Parkinsons Disease")
#############################################