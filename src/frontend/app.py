import streamlit as st
import requests

st.title('GAN for Image Colorization')
# st.markdown('Xander Gallegos')

# st.divider()

col1, col2 = st.columns([0.8,0.2])

with col1:
    input_file = st.file_uploader('Upload your image', accept_multiple_files=False)

with col2:
    predict = st.button('Predict')


if predict:
    url = 'http://gan-api:5051/api/v1/colorize/'
    headers = {'content-type': 'application/json'}

    file_bytes = input_file.getvalue()
    response = requests.post(
        url=url,
        files={'file': file_bytes},
        headers=headers
    )

    col3, col4 = st.columns(2)

    with col3:
        st.image(input_file.getvalue())

    with col4:
        st.image(response)
    # st.subheader(response.json()['prediction'])
