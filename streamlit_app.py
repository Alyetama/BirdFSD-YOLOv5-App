import mimetypes

import requests
import streamlit as st
from PIL import Image


def main():
    st.title('BirdFSD-YOLOv5')
    model = requests.get('https://api.aibird.me/model',
                         data='{"version": "latest"}')
    model_info = model.json()

    st.markdown(f'Version: `{model_info["name"]}-v{model_info["version"]}`')

    uploaded_files = st.file_uploader('Choose an image file',
                                      accept_multiple_files=True,
                                      type=['png', 'jpg', 'jpeg'])

    col1, col2 = st.columns(2)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            data = f'{{"file": {uploaded_file.getvalue()}}}'
            col1.image(Image.open(uploaded_file))
            fname = uploaded_file.name
            fdata = uploaded_file.getvalue()
            content_type = mimetypes.guess_type(uploaded_file.name)[0]

            res = requests.post('https://api.aibird.me/predict',
                                files={'file': (fname, fdata, content_type)})
            result = res.json()
            col2.image(result['results']['labeled_image_url'])
            with st.expander(f'Results ({uploaded_file.name})'):
                st.json(result)


if __name__ == '__main__':
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
