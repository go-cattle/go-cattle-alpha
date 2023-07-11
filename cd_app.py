## Import required libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
## import functions from support.py
from support import * 
from PIL import Image

st.set_page_config(
    page_title="Go-Cattle // Cattle Healthcare",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

df = pd.read_csv('dataset/augean.csv')
df_s  = df.drop(columns=['Disease'])
# Calculate the sum of each column
column_sums = df_s.sum()
del df_s
# Select the top ten columns with the highest sums
top_columns = column_sums.nlargest(10).index.tolist()

## Defining two tabs
tab1,tab2,tab3,tab4,tab5 = st.tabs(["Home        ","Prediction  ","Diseases      ","Feedback     ", "Legal"])
## Style Section
with open("style.css") as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
## Home tab
with tab1:
    st.markdown(f"<span style='color: #1fd655; font-weight: bold; font-size: 25px; margin: auto; text-align: center;'>Go-Cattle</span>", unsafe_allow_html=True)
    st.markdown(f"<p style='font-weight: 600; line-height: 3px; text-decoration: underline;'>A Cattle Healthcare Platform</p>", unsafe_allow_html=True)
    image=Image.open('gcb.jpg')
    st.image(image)
    st.markdown('### :red[**What is Go Cattle?**]')
    st.markdown("Go Cattle is a :green[**Cattle Healthcare Platform**]. India is the Home to about 17% of the World's Cows and For Every 1 Registered Vet in India, There are about 50,000 cows. Due To These Reasons, About 65% of The Cows Cannot Get Proper Healthcare and Treatments. It is Very Important to increase awareness about this topic because This Leads to Thousands if Not Hundreds of Thousands of Cattle Dying Every Year.")
    st.markdown("Go Cattle Provides a Variety of Resources for The Welfare of Cattles. One of The Main Features is an advanced web application designed to **analyze** :red[diseases] in cattle based on the :yellow[**symptoms**] provided. With its cutting-edge ML-model analyzer, Go Cattle ensures accurate and efficient diagnosis, empowering cattle owners and veterinarians to make informed decisions about their livestock's health.")
    st.markdown("Our ML-model boasts an outstanding :green[**accuracy rate of 95%+**], surpassing the required medical standards" f"<sup>#</sup>" " Developed using a vast dataset of *20,499 parameters* sourced from reliable and up-to-date information gathered through web crawling & web scraping, Go Cattle provides a robust foundation for precise disease identification.", unsafe_allow_html=True)
    st.markdown("Equipped with an extensive range of 123 unique symptoms and a comprehensive list of 163 unique diseases, Go Cattle covers a wide spectrum of ailments that can affect cattle. By inputting the observed symptoms, the system swiftly processes the information and generates a reliable diagnosis, enabling prompt action to be taken. :violet[ *The Dataset has been gone through Vigorous Changes Recently and There's A High Possibility that our team might have messed up something in the Process (as of 10th July 2023)*]")
    
    
    col_1,col_2  = st.columns(2)

    with col_1:
        for item in top_columns[:5]:
            st.markdown(f"<span style='color: blue; font-weight: bold;'>{item}</span>", unsafe_allow_html=True)
    with col_2:
        
        for item in top_columns[5:]:
            st.markdown(f"<span style='color: blue; font-weight: bold;'>{item}</span>", unsafe_allow_html=True)


    symptoms = df.columns
    symptoms2 = symptoms[1:]
    symptoms = sorted(symptoms[1:])

    del df

    with st.sidebar:
        st.title(":green[List of Symptoms]")
        highlighted_elements = highlight_list_elements(symptoms)
        # for symptom in symptoms:
        #     st.markdown(f"<span style='color: blue; font-weight: bold; font-size: 15px;'>{symptom}</span>",unsafe_allow_html=True)
        selected_items = []
        for item in symptoms:
            checkbox = st.checkbox(item)
            if checkbox:
                selected_items.append(item)

        # st.markdown(highlighted_elements, unsafe_allow_html=True)



## Predictions Tab
with tab2:
    #model part
    # Initialize the Random Forest Classifier
    classifier = RandomForestClassifier()
    st.markdown("## :green[Go-Cattle's ]:orange[ Disease Analyzer]")
    model = joblib.load('model.pkl')

    ## Select symtoms part
    st.markdown('### :green[**Select**] :red[**Symptoms:**]')
    pred_symptoms=st.multiselect(label=" ", options=symptoms2)
    pred_symptoms.extend(selected_items)
    st.markdown(f"<span style='color: #1fd655; font-weight: 600; font-size: 19px;'>Selected Symtomps are: </span><span style='color: yellow; font-weight: 500; font-size: 18px;' >{', '.join(pred_symptoms)}</span>",unsafe_allow_html=True)
    
    pred_df = pd.DataFrame(0,index=[0],columns=symptoms2)
    pred_df[pred_symptoms]=1
    # st.table(pred_df)
    result = model.predict(pred_df)
    st.subheader(f"Most probable Disease: :red[{result[0]}]")
    top_five = model.predict_proba(pred_df)
    # Get the classes from the random forest model
    classes = model.classes_
    # Get the indices of the top five classes based on probability
    top_class_indices = np.argsort(top_five, axis=1)[:, -5:]
    # Get the top five classes and their probabilities
    top_classes = classes[top_class_indices][:, ::-1]
    top_probabilities = np.take_along_axis(top_five, top_class_indices, axis=1)[:, ::-1]

    


    # disaplying the Top diseses part
    data = []
    for i, row in enumerate(top_classes):
        input_row = []
        for j, cls in enumerate(row):
            probability = top_probabilities[i][j]
            input_row.append((cls, probability))
        data.append(input_row)

    n_dis = 3
    column_names = [f'Top {i+1} Class' for i in range(5)] + [f'Top {i+1} Probability' for i in range(5)]
    # st.subheader(f"List of most probable Deseases ")
    
    df3 = pd.DataFrame(data[0],columns = ['Disease','Probability(%)'])
    df3['Probability(%)'] = df3['Probability(%)']*100
    df3.index = range(1, len(df3) + 1)

    col1,col2 = st.columns([3,1])

    with col1:
        st.text("\n")
        st.markdown(f"<span style='color: #; font-weight: bold; font-size: 25px;'>Most Probable Diseases</span>",unsafe_allow_html=True)
    with col2:
        st.text("\n")
        n_dis = st.selectbox(" ",(3,5))

    st.dataframe(df3.head(n_dis),1000)
    # st.dataframe(st.dataframe(df3.style.highlight_max(axis=0)))
    
with tab3:
    st.title("__Diseases__")
    #populate all of the diseases from the dataset
    st.markdown("## :orange[__***Work in Progress***__]")

with tab4:
    st.subheader("Feedback Portal")
    colFB1,colFB2 = st.columns([1,3])
    with colFB1:
        name  = st.text_input("Enter your name")
        email = st.text_input("Enter your email id")
    with colFB2:
        feedback = st.text_area("Enter your feedback here")
    

    if st.button("Submit"):
        if (feedback and name and email):
            # Save feedback as a file
            filename = save_feedback(name,email,feedback)
            st.success(f"Thank you for your feedback! Your feedback has been saved.")
        else:
            st.warning("Please enter your details before submitting.")

with tab5:
    st.title("Legal")
    st.markdown("### :green[**Terms of Service**]")
    st.markdown("""Disclaimer: 

The advice or results generated by the Go Cattle app are derived from an artificial intelligence machine learning model. While efforts are made to ensure accuracy levels of 95% or higher, it is crucial to note that these outcomes may be subject to inaccuracies and should not be regarded as medical information. Therefore, the advice provided by the app should never be solely relied upon without seeking the guidance of a qualified veterinarian. 

It is important to understand that:

1. The Go Cattle app is not a substitute for professional veterinary care.
2. The results obtained from the app should not be used as a means to diagnose or treat any medical condition.
3. If you have concerns about the health of your cattle, it is imperative that you consult a veterinarian without delay.

By utilizing the Go Cattle app, you expressly acknowledge and agree that:

1. Go Cattle shall not be held liable or accountable for any mishaps, damages, injuries, or losses arising from the use of the advice or results provided by the app.
2. The app's advice and results are not a replacement for personalized veterinary care and should be considered as supplementary information only.

We hope this disclaimer serves to clarify the limitations of the Go Cattle app and the need for professional veterinary consultation. Your understanding and compliance with these terms are greatly appreciated.

Thank you for choosing and using Go Cattle!""")