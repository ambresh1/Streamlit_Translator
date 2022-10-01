import streamlit as st

import docx
def getText(filename):
    doc = docx.Document(filename)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)




st.title('Translator App')
st.markdown("Translate from Docx file")
st.sidebar.subheader("File Upload")

datas=st.sidebar.file_uploader("Original File")
# data=getText("C:\Users\Ambresh C\Desktop\Python Files\Translators\Trail Doc of 500 words.docx")

# st.sidebar.download_button(label='Download Translated File',file_name='Translated.docx')
st.write(getText(datas))
