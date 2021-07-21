# https://www.tutorialspoint.com/flask
import joblib
import numpy as np
#from sklearn.externals import joblib
import os
import pickle
import pandas as pd
import streamlit as st
from MyEncoder import My_encoder
#from category_encoders import one_hot
binary=lambda val: 1 if val == "Yes" else 0
gender=lambda val: "M" if val == "Male" else "F"
quarter=lambda val: val.split("-")[1]
year=lambda val: int(val.split("-")[0])
st.title('Drug Marketing and Physician Targeting')
base_dir=os.path.abspath(os.path.curdir)


st.markdown("""**Business Problem:** A Pharma Company had launched a drug; however, some physicians are yet to 
prescribe it for the first time. A key client stakeholder has reached out to a Decision 
Sciences Principal in Axtria for help to identify potential physicians who are most likely 
to start prescribing the drug in the next quarter in order to channelize the marketing 
efforts more effectively while targeting potential physicians.""")
# @st.cache()
# def load_pickle():
#
#     return all_feature,one_dump,scaler,model

st.markdown("**Note:** Submit the below form data to predict pysician who will likely to prescribe the drug in next quarter")

year_quarter=st.selectbox("Select Quarter",
                             options=["2019-Q3", "2019-Q4", "2020-Q1","2020-Q2", "2020-Q3"])

brand_prescribed=st.selectbox("brand_prescribed",
                             options=["Yes","No"])
total_representative_visits=st.slider("total_representative_visits", min_value=0, max_value=55, step=1)
total_sample_dropped=st.slider("total_sample_dropped", min_value=0, max_value=1392, step=1)
physician_hospital_affiliation=st.number_input("physician_hospital_affiliation", min_value=0, max_value=1, step=1)

physician_in_group_practice=st.selectbox("physician_in_group_practice",
                             options=["Yes","No"])
total_prescriptions_for_indication1=st.slider("total_prescriptions_for_indication1", min_value=0, max_value=2029, step=1)
total_prescriptions_for_indication2=st.slider("total_prescriptions_for_indication2", min_value=0, max_value=2932, step=1)
total_patient_with_commercial_insurance_plan=st.slider("total_patient_with_commercial_insurance_plan", min_value=0, max_value=2109, step=1)
total_patient_with_medicare_insurance_plan=st.slider("total_patient_with_medicare_insurance_plan", min_value=0, max_value=4746, step=1)
total_patient_with_medicaid_insurance_plan=st.slider("total_patient_with_medicaid_insurance_plan", min_value=0, max_value=2538, step=1)
total_competitor_prescription=st.slider("total_competitor_prescription", min_value=0, max_value=8815, step=1)
new_prescriptions=st.slider("new_prescriptions", min_value=0, max_value=3790, step=1)
physician_gender=st.selectbox("Select Gender",
                             options=["Male","Female"])
physician_speciality=st.selectbox("Select Speciality",
                             options=["nephrology", "urology", "other"])

submit = st.button("Submit")

if submit:
    # all_feature, one_dump, scaler, model = load_pickle()



    model = joblib.load(base_dir + '/data/LGR_model.pkl')

    data = [{'year_quarter': year_quarter, 'brand_prescribed': binary(brand_prescribed), 'total_representative_visits':total_representative_visits,
             'total_sample_dropped': total_sample_dropped,"saving_cards_dropped": np.random.randint(0,140),"vouchers_dropped":np.random.randint(0,116),
             'total_seminar_as_attendee': np.random.randint(0,5),"total_seminar_as_speaker": np.random.randint(0,42),
             'physician_hospital_affiliation': physician_hospital_affiliation, 'physician_in_group_practice':binary(physician_in_group_practice),
             'total_prescriptions_for_indication1': total_prescriptions_for_indication1, 'total_prescriptions_for_indication2': total_prescriptions_for_indication2,
             'total_prescriptions_for_indication3': np.random.randint(0,2000),
             'total_patient_with_commercial_insurance_plan':total_patient_with_commercial_insurance_plan,
             'total_patient_with_medicare_insurance_plan': total_patient_with_medicare_insurance_plan, 'total_patient_with_medicaid_insurance_plan': total_patient_with_medicaid_insurance_plan,

             'brand_web_impressions':np.random.randint(0,500),
            'brand_ehr_impressions':np.random.randint(0,800),
            'brand_enews_impressions':np.random.randint(0,48),
            'brand_mobile_impressions':np.random.randint(0,140),
            'brand_organic_web_visits':np.random.randint(0,2),
            'brand_paidsearch_visits':np.random.randint(0,2),
             'total_competitor_prescription': total_competitor_prescription,
             'new_prescriptions': new_prescriptions,
             'urban_population_perc_in_physician_locality':np.random.randint(0,2),
                                                        'percent_population_with_health_insurance_in_last10q':np.random.randint(0,2),
             'physician_tenure':np.random.randint(5,60),
            'physician_age':np.random.randint(28,90),
             'physician_gender': gender(physician_gender), 'physician_speciality':physician_speciality,"quarter":quarter(year_quarter),"year":year(year_quarter)}]
    #st.write(pd.DataFrame(data))   
    xq=pd.DataFrame(data)
    # print(xq.columns.values)
    categorical_columns = ["physician_gender", "physician_in_group_practice", "physician_hospital_affiliation"
        , "physician_speciality", "brand_prescribed", "year_quarter", "quarter", "year"]
    num_cols = ['total_representative_visits',
                'total_sample_dropped',
                'saving_cards_dropped',
                'vouchers_dropped',
                'total_seminar_as_attendee',
                'total_seminar_as_speaker',
                'total_prescriptions_for_indication1',
                'total_prescriptions_for_indication2',
                'total_prescriptions_for_indication3',
                'total_patient_with_commercial_insurance_plan',
                'total_patient_with_medicare_insurance_plan',
                'total_patient_with_medicaid_insurance_plan',
                'brand_web_impressions',
                'brand_ehr_impressions',
                'brand_enews_impressions',
                'brand_mobile_impressions',
                'brand_organic_web_visits',
                'brand_paidsearch_visits',
                'total_competitor_prescription',
                'new_prescriptions',
                'urban_population_perc_in_physician_locality',
                'percent_population_with_health_insurance_in_last10q',
                'physician_tenure',
                'physician_age']
    # print("shape======",xq.shape)
    filename = os.path.join(base_dir, 'data', 'ohe_dump2.pkl')
    filename = open(filename, 'rb')
    one_dump = pickle.load(filename)
    xq_cat = one_dump.transform(xq[categorical_columns])
    # scaler.clip = False
    filename = os.path.join(base_dir, 'data', 'scaler_dump2.pkl')
    filename = open(filename, 'rb')
    scaler = pickle.load(filename)
    xq_num=scaler.transform(xq[num_cols])
    xq_data=np.hstack((xq_cat,xq_num))
    filename = os.path.join(base_dir, 'data', 'all_features2.pkl')
    filename = open(filename, 'rb')
    all_feature = pickle.load(filename)
    xq_data=pd.DataFrame(xq_data,columns=all_feature)

    #y_pred = rf_model.predict(xq_point_new)
    y_pred = model.predict(xq_data)
    print(y_pred)
    if (y_pred == [0]):
        y_pred_new='CLASS-1-LOW'
    elif (y_pred == [1]):
        y_pred_new='CLASS-2-MEDIUM'
    elif (y_pred == [2]):
        y_pred_new='CLASS-3-HIGH'
    else: 
        y_pred_new='CLASS-4-VERY_HIGH'
    #print('Predicted Text for xq point: ',y_pred_new)
    st.write("Classifying the point.....")
    st.success(y_pred_new)
