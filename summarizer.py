import streamlit as st
from io import StringIO

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

st.header("""Document Summarizer""")
uploaded_file = st.file_uploader("Choose a word file")



if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    string_data = stringio.read()

    template = """Read the information {string_data} and I want you to create 
    1. Headline 
    2. Summary"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["string_data"],
    template=template,
    )

    llm = ChatOpenAI(temperature =0,openai_api_key='Add your OpenAI api key')
    chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
    res=chain.invoke(input={'string_data':string_data})
    st.write(res)