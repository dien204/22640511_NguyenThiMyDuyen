import joblib
import pandas as pd
import streamlit as st


bundle = joblib.load("models/model.pkl")
model = bundle["model"]
feature_info = bundle["feature_info"]
task_type = bundle["task_type"]

st.title("MLOps Demo")
st.write("Nhập dữ liệu đầu vào để dự đoán.")

input_data = {}

for col, info in feature_info.items():
    if info["type"] == "numeric":
        input_data[col] = st.number_input(col, value=float(info["default"]))
    else:
        options = info.get("options", [])
        if not options:
            options = [info.get("default", "")]
        input_data[col] = st.selectbox(col, options=options)

if st.button("Dự đoán"):
    X = pd.DataFrame([input_data])
    prediction = model.predict(X)[0]

    if task_type == "regression":
        st.success(f"Giá trị dự đoán: {prediction}")
    else:
        st.success(f"Kết quả phân loại: {prediction}")
