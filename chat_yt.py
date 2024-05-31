from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_community.vectorstores import chroma
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.llms import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
import streamlit as st
import re
# import chromadb

def is_valid_youtube_url(url):
    # Regular expression to match YouTube video URLs
    youtube_regex = (
        r'(https?://)?(www\.)?'
        '(youtube|youtu|youtube-nocookie)\.(com|be)/'
        '(watch\?v=|embed/|v/|.+\?v=)?([^&=%\?]{11})')

    # Match the provided URL with the regex pattern
    match = re.match(youtube_regex, url)
    
    # If there's a match, it's a valid YouTube URL
    if match:
        return True
    else:
        return False


def load_documents(link):

    if is_valid_youtube_url(link)==False:
        return None
    loader = YoutubeLoader.from_youtube_url(
        link,
        add_video_info=True,
        language=["en","hi","id"],
        translation="en",
    )
    documents = loader.load()
    return documents

def split_documents(documents):
    # st.write(documents)
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "",
        ],
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    # Make splits
    splits = text_splitter.split_documents(documents)
    return splits

# create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
def define_vectore_store(splits):

    # load it into FAISS
    db = FAISS.from_documents(documents=splits, embedding=embedding_function)
    return db

# def summarize_rag_chain(splits):
#     # llm
#     HUGGINGFACEHUB_API_TOKEN = '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'
#     repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
#     llm = HuggingFaceHub(repo_id=repo_id,model_kwargs={"temperature":0.5, "max_length":1024},huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

#     # summarize_chain = load_summarize_chain(llm=llm,chain_type='refine')
#     # summary = summarize_chain.run(splits)
#     # return summary
#     prompt_template = """Write a concise summary of the following:
#     {text}
#     CONCISE SUMMARY:"""
#     prompt = PromptTemplate.from_template(prompt_template)

#     refine_template = (
#         "Your job is to produce a final summary\n"
#         "We have provided an existing summary up to a certain point: {existing_answer}\n"
#         "We have the opportunity to refine the existing summary"
#         "(only if needed) with some more context below.\n"
#         "------------\n"
#         "{text}\n"
#         "------------\n"
#         "Given the new context, refine the original summary in Italian"
#         "If the context isn't useful, return the original summary."
#     )
#     refine_prompt = PromptTemplate.from_template(refine_template)
#     chain = load_summarize_chain(
#         llm=llm,
#         chain_type="stuff",
#         # question_prompt=prompt,
#         # refine_prompt=refine_prompt,
#         # return_intermediate_steps=True,
#         # input_key="input_documents",
#         # output_key="output_text",
#     )
#     result = chain.run(splits)
#     return result

def retriever_chain(db):
    retriever = db.as_retriever()

    # Prompt
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # llm
    HUGGINGFACEHUB_API_TOKEN = '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'
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
    llm = HuggingFaceHub(repo_id=repo_id,model_kwargs={"temperature":0.5, "max_length":1024},huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN)

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

def main():
    st.title('Chat with youtube video')

    # Sidebar for YouTube URL input
    st.sidebar.header("YouTube URL")
    youtube_url = st.sidebar.text_input("Enter YouTube URL")

    # Process URL button
    if st.sidebar.button("Process URL"):
        if youtube_url:
            st.sidebar.write("Processing URL:", youtube_url)
            if is_valid_youtube_url(youtube_url):
                st.sidebar.write('URL is valid')
            else:
                st.sidebar.write('URL is invalid')
    
    question = st.text_input('Enter your query')
    if st.button("Ask Question"):
        if question:
            documents = load_documents(youtube_url)
            split = split_documents(documents)
            db = define_vectore_store(split)
            rag_chain = retriever_chain(db)

            answer = rag_chain.invoke(question)
            idx = answer.find('Question:')
            answer = answer[idx:]
            st.write(answer)
        else:
            st.warning("Please enter a question")

if __name__=='__main__':
    main()