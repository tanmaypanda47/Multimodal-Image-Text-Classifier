import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

from src.inference import predict


st.set_page_config(
    page_title="Multimodal Product Classifier",
    page_icon="🛍️",
    layout="wide"
)



st.title("🛍️ Multimodal Product Classifier")

st.markdown("""
### Classify Fashion Products Using AI

This model uses:

- 📷 Product Images
- 📝 Product Titles
- 🤖 CLIP Multimodal Embeddings
- ⚡ GPU Accelerated Inference
""")

st.divider()



st.sidebar.title("📊 Model Information")

st.sidebar.info("""
**Model:** CLIP ViT-B/32

**Dataset:** Fashion Product Images

**Classes**

- Apparel
- Accessories
- Footwear
- Personal Care
- Free Items
""")



col1, col2 = st.columns(2)

with col1:

    uploaded_file = st.file_uploader(
        "Upload Product Image",
        type=["jpg", "jpeg", "png"]
    )

with col2:

    product_title = st.text_input(
        "Product Title",
        placeholder="Nike Running Shoes"
    )



if uploaded_file is not None and product_title:

    try:

        image = Image.open(
            uploaded_file
        ).convert(
            "RGB"
        )

        st.image(
            image,
            caption="Uploaded Product",
            width=300
        )

        with st.spinner(
            "Analyzing Product..."
        ):

            predictions = predict(
                image,
                product_title
            )

        st.success(
            "Prediction Complete"
        )

        top_prediction = predictions[0]

    

        card1, card2 = st.columns(2)

        with card1:

            st.markdown(f"""
            <div style="
                background-color:#f8f9fa;
                padding:25px;
                border-radius:15px;
                border-left:8px solid #4CAF50;
                box-shadow:0px 2px 10px rgba(0,0,0,0.1);
            ">
                <h4 style="color:#555;">
                    Predicted Category
                </h4>
                <h1 style="color:#111;">
                    {top_prediction['category']}
                </h1>
            </div>
            """,
            unsafe_allow_html=True)

        with card2:

            st.markdown(f"""
            <div style="
                background-color:#f8f9fa;
                padding:25px;
                border-radius:15px;
                border-left:8px solid #2196F3;
                box-shadow:0px 2px 10px rgba(0,0,0,0.1);
            ">
                <h4 style="color:#555;">
                    Confidence
                </h4>
                <h1 style="color:#111;">
                    {top_prediction['confidence']}%
                </h1>
            </div>
            """,
            unsafe_allow_html=True)

        st.write("")

       

        st.subheader("Confidence Score")

        st.progress(
            float(
                top_prediction["confidence"]
            ) / 100
        )

        st.write(
            f"{top_prediction['confidence']}%"
        )

        if top_prediction["confidence"] >= 90:

            st.success(
                "High Confidence Prediction"
            )

        elif top_prediction["confidence"] >= 70:

            st.warning(
                "Moderate Confidence Prediction"
            )

        else:

            st.error(
                "Low Confidence Prediction"
            )

        st.divider()

      
        st.subheader(
            "Top 3 Predictions"
        )

        df = pd.DataFrame(
            predictions
        )

        fig = px.bar(

            df,

            x="category",

            y="confidence",

            text="confidence",

            title="Prediction Confidence Distribution"
        )

        fig.update_traces(
            textposition="outside"
        )

        fig.update_layout(
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.dataframe(
            df,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Prediction Error: {str(e)}"
        )



st.divider()

st.markdown("""
### 🚀 Features

✅ Multimodal Learning (Image + Text)

✅ CLIP Transformer Architecture

✅ Confidence Scores

✅ Top-3 Predictions

✅ Interactive Visualization

✅ GPU Accelerated Inference

---


""")