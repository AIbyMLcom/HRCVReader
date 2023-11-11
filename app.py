import dotenv

dotenv.find_dotenv()
dotenv.load_dotenv()

import gradio as gr
import sys
import os
import argparse
import src.color as color

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.prompts import ChatPromptTemplate
from src.openai import embeddings, chat_llm

DOCUMENT_BASE = "./docs"


def init_and_create_store() -> Chroma:
    documents = []
    for file in os.listdir("docs"):
        if file.endswith(".pdf"):
            pdf_path = os.path.join(DOCUMENT_BASE, file)
            loader = PyPDFLoader(pdf_path)
            documents.extend(loader.load())
        elif file.endswith(".docx") or file.endswith(".doc"):
            doc_path = os.path.join(DOCUMENT_BASE, file)
            loader = Docx2txtLoader(doc_path)
            documents.extend(loader.load())
        elif file.endswith(".txt"):
            text_path = os.path.join(DOCUMENT_BASE, file)
            loader = TextLoader(text_path, encoding="utf-8")
            documents.extend(loader.load())
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=300, chunk_overlap=50
    )
    documents = text_splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        documents, embedding=embeddings, persist_directory="./data"
    )
    vectordb.persist()
    return vectordb


def ask_gpt(question: str, prompt: str, vector_store: Chroma) -> tuple[str, str]:
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vector_store.as_retriever(), llm=chat_llm
    )
    manuals = retriever_from_llm.get_relevant_documents(question)
    information = "\n".join([manual.page_content for manual in manuals])

    question_chain = LLMChain(
        llm=chat_llm,
        prompt=ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    "{prompt} Here are the extracted information:\n\n```{extracted_context}```\n\nQuestion: {question}",
                ),
                ("ai", "Response:"),
            ]
        ),
        verbose=False,
        output_key="answer",
    )

    return information, question_chain.run(
        question=question,
        prompt=prompt,
        extracted_context=information,
    )


def init_chatbot_handler(vector_store: Chroma):
    def handler(message: str, _: list[list[str]]):
        _, result = ask_gpt(message, "", vector_store)
        return result

    return handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Langchain Question Answer Demo.")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["web", "cli"],
        help="Select mode web or cli",
        default="cli",
    )
    args = parser.parse_args()

    vectordb = init_and_create_store()
    chat_history = []

    if args.mode == "cli":
        print(
            f"{color.yellow}---------------------------------------------------------------------------------"
        )
        print(
            "Langchain Question Answer Demo. You are now ready to start interacting with your documents"
        )
        print(
            "---------------------------------------------------------------------------------"
        )
        while True:
            query = input(f"{color.green}Prompt: ")
            if query == "exit" or query == "quit" or query == "q" or query == "f":
                print("Exiting")
                sys.exit()
            if query == "":
                continue
            _, result = ask_gpt(query, "", vectordb)
            print(f"{color.white}Answer: " + result)
            chat_history.append((query, result))

    else:
        CSS = """
        .contain { display: flex; flex-direction: column; }
        .gradio-container { height: 100vh !important; }
        #component-0 { height: 100%; }
        #component-3 { flex-grow: 1; overflow: auto;}
        """

        message_handler = init_chatbot_handler(vectordb)
        demo = gr.ChatInterface(
            message_handler,
            title="Langchain Question Answer Demo",
            css=CSS,
        )
        demo.launch(share=True)
