import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Mental Health Screening Tool", page_icon="🧠", layout="centered")

st.title("🧠 Mental Health Risk Screening Tool")
st.write("Complete the 20-question survey below to evaluate mental health risk.")

# Load the trained Decision Tree model
model_path = os.path.join(os.path.dirname(__file__), 'final_decision_tree_model.pkl')

if not os.path.exists(model_path):
    st.error("Model file 'final_decision_tree_model.pkl' not found. Please train the model in Decision_Tree.ipynb first.")
else:
    model = joblib.load(model_path)

    # -------------------------------------------------------------
    # SECTION 1: Daily Mood & Stress Symptoms (Scale 1 to 10)
    # -------------------------------------------------------------
    st.header("1. Daily Mood & Stress Symptoms (Rate 1 to 10)")
    col1, col2 = st.columns(2)

    with col1:
        work_stress = st.slider("Work / Study Stress", 1, 10, 5)
        financial_stress = st.slider("Financial Stress", 1, 10, 5)
        feeling_sad = st.slider("Feeling Sad or Down", 1, 10, 4)
        anxious = st.slider("Feeling Anxious or Nervous", 1, 10, 4)
        loneliness = st.slider("Loneliness / Isolation", 1, 10, 4)
        sleep_trouble = st.slider("Sleep Trouble", 1, 10, 5)
        fatigue = st.slider("Fatigue / Low Energy", 1, 10, 5)

    with col2:
        feeling_worthless = st.slider("Feeling Worthless / Low Self-Esteem", 1, 10, 4)
        irritability = st.slider("Irritability / Short Temper", 1, 10, 4)
        concentration = st.slider("Difficulty Concentrating", 1, 10, 4)
        appetite = st.slider("Poor Appetite or Overeating", 1, 10, 4)
        compulsive = st.slider("Compulsive / Repetitive Habits", 1, 10, 4)
        obsessive = st.slider("Obsessive / Intrusive Thoughts", 1, 10, 4)

    # -------------------------------------------------------------
    # SECTION 2: Protective Factors (Scale 1 to 10)
    # -------------------------------------------------------------
    st.header("2. Protective Factors (Rate 1 to 10)")
    pcol1, pcol2 = st.columns(2)
    with pcol1:
        social_support = st.slider("Social Support (Friends, Family, Community)", 1, 10, 6)
    with pcol2:
        job_satisfaction = st.slider("Job / College Satisfaction", 1, 10, 6)

    # -------------------------------------------------------------
    # SECTION 3: Clinical & Medical History (Yes / No)
    # -------------------------------------------------------------
    st.header("3. Clinical & Medical History")
    hcol1, hcol2 = st.columns(2)
    with hcol1:
        family_history = st.checkbox("Family History of Mental Illness")
        trauma_history = st.checkbox("History of Psychological Trauma")
        on_medication = st.checkbox("Currently on Medication for Stress/Anxiety")
    with hcol2:
        self_harm = st.checkbox("Past or Present Thoughts of Self-Harm")
        suicidal = st.checkbox("Past or Present Suicidal Thoughts")

    st.write("---")

    # -------------------------------------------------------------
    # CALCULATE SCORES & PREDICT
    # -------------------------------------------------------------
    if st.button("Evaluate Mental Health Risk", type="primary"):
        # 1. Sum up the 13 mood symptoms
        symptoms_sum = (
            work_stress + financial_stress + feeling_sad + anxious +
            loneliness + sleep_trouble + fatigue + feeling_worthless +
            irritability + concentration + appetite + compulsive + obsessive
        )

        # 2. Compute Total Distress Score (minus protective buffers)
        total_distress = symptoms_sum - social_support - job_satisfaction

        # 3. Compute Severe Clinical Risk Score
        severe_risk = (
            (1 if suicidal else 0) * 3 +
            (1 if self_harm else 0) * 3 +
            (1 if family_history else 0) * 2 +
            (1 if trauma_history else 0) * 2 +
            (1 if on_medication else 0)
        )

        # 4. Compute Overall Risk Index
        overall_risk = total_distress + (severe_risk * 5)

        # Prepare 4-feature dataframe for the model
        input_df = pd.DataFrame([{
            'Overall_Risk_Index': overall_risk,
            'Total_Distress_Score': total_distress,
            'Severe_Risk_Score': severe_risk,
            'Social_Support': social_support
        }])

        prediction = model.predict(input_df)[0]

        # Display results
        st.subheader("Screening Result:")
        st.write(f"• **Overall Risk Index:** **{overall_risk:.1f}** (Cutoff threshold is **54.50**)")
        st.write(f"• **Daily Distress Score:** {total_distress:.1f} | **Severe Clinical Flags Score:** {severe_risk:.1f}")

        if prediction == 0:
            st.success("🟢 **Healthy / Low Risk (Class 0)**")
            st.write("The assessment indicates healthy psychological functioning with adequate protective factors.")
        else:
            st.warning("🔴 **Elevated Risk / Needs Attention (Class 1)**")
            st.write("The assessment indicates elevated psychological distress or clinical risk indicators. Professional consultation or counseling support is recommended.")
