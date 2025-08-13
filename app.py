import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader

## Streamlit App
st.set_page_config(page_title="Langchain: Summarize Text From YT or Website")
st.title("Langchain: Summarize Text From YT or Website")
st.subheader('Summarize URL')


## Get the Groq API Key and URL(YT or website) field to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

generic_url = st.text_input("URL",label_visibility="collapsed")


## Gemma Model Using Groq API
llm = ChatGroq(model="Llama3-8b-8192",groq_api_key=groq_api_key)

prompt_template = """
Provide the summary of the following content in 300 words:
Content:{text}
"""

prompt = PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url): 
        st.error("Please enter a valid URL. It can may be a YT url or website url")
    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or YT Video data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0"})
                docs = loader.load()

                ## Chain for Summarization
                chain = load_summarize_chain(llm, chain_type="stuff",prompt=prompt)
                output_summary = chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
