import io
import os

import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image


def main():
    st.title('BirdFSD-YOLOv5')
    st.markdown('Version: `BirdFSD-YOLOv5-v1.0.0-alpha.1.2`')

    model = torch.hub.load('ultralytics/yolov5',
                           'custom',
                           path=os.environ['WEIGHTS'])

    uploaded_files = st.file_uploader('Choose an image file',
                                      accept_multiple_files=True,
                                      type=['png', 'jpg', 'jpeg'])

    col1, col2 = st.columns(2)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            col1.image(image)
            model_preds = model(image)
            col2.image(model_preds.render()[0])
            with st.expander('Bounding box [x, y, w, h], class & score'):
                for res in model_preds.xywhn[0]:
                    x, y, w, h, score, n = res.tolist()
                    st.code(
                        f'Bb: {[round(z, 6) for z in [x, y, w, h]]} | Label: {model.names[int(n)]} | Score: {round(score, 2)}'
                    )


if __name__ == '__main__':
    load_dotenv()
    st.set_page_config(page_title='BirdFSD-YOLOv5',
                   page_icon='🐦',
                   layout='wide',
                   initial_sidebar_state='expanded')
    st.markdown("""<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .css-e370rw {visibility: hidden;}
    #plot > div:nth-child(1) > a:nth-child(1) {visibility: hidden;}
    .viewerBadge_link__1S137 {visibility: hidden;}
    </style>""",
                unsafe_allow_html=True)
    main()
