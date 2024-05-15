import os
from pathlib import Path
from dotenv import dotenv_values

from langchain import hub
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from typing import List
from PIL import Image


def update_env_vars(env_file_path: str=None):
    if env_file_path:
        env_config = dotenv_values(env_file_path)
        os.environ = {**os.environ, **env_config}

class RAG:
    def __init__(self):
        self.encoder = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")


    def query_links(self, links: List[str], query: str) -> str:
        vectorstore = self._get_vectorstore_from_links(links)
        rag_chain = self.get_rag_chain(vectorstore)
        answer = rag_chain.invoke(query)
        vectorstore.delete_collection()
        return answer


    def query_images(self, image_path_list: List[str], query: str) -> str:
        raw_images = [Image.open(img_path).convert('RGB') for img_path in image_path_list]
        vectorstore = self._get_vectorstore_from_images(raw_images, image_path_list)
        rag_chain = self.get_rag_chain(vectorstore)
        answer = rag_chain.invoke(query)
        vectorstore.delete_collection()
        return answer


    def get_rag_chain(self, vectorstore):
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
            {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain


    def _get_vectorstore_from_links(self, links):
        # Use this to ask GPT about content on websites
        import bs4
        from langchain_community.document_loaders import WebBaseLoader

        docs = []
        for link in links:
            # Load, chunk and index the contents of the blog.
            loader = WebBaseLoader(
                web_paths=(link,),
                # TODO: we can add constrains based on the website type
                # bs_kwargs=dict(
                #     parse_only=bs4.SoupStrainer(
                #         class_=()
                #     )
                # ),
            )
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=splits, embedding=self.encoder)

        return vectorstore
    

    def _get_vectorstore_from_images(self, raw_images_list: List[Image.Image], raw_images_paths: List[str]=None):
        # Use this to ask GPT about content in images
        descriptions = self._extract_contest_from_images(raw_images_list)

        documents = []
        for i, description in enumerate(descriptions):
            source = raw_images_paths[i] if raw_images_paths != None else f"image_{i}"
            description = "{" + f"source: '{source}', description: '{description}'" + "}"
            print(description)
            doc = Document(page_content=description, metadata={"source": source})
            documents.append(doc)
        
        vectorstore = Chroma.from_documents(documents=documents, embedding=self.encoder)
        return vectorstore


    def _get_vectorstore_from_videos(self, videos_path_list, use_transcription=False):
        # Use this to ask GPT about content in videos
        pass
    

    def _extract_contest_from_images(self, raw_images: List, model_type="Salesforce/blip-image-captioning-large"):
        from transformers import BlipProcessor, BlipForConditionalGeneration

        descriptions = []
        
        processor = BlipProcessor.from_pretrained(model_type)
        model = BlipForConditionalGeneration.from_pretrained(model_type)

        # conditional image captioning
        for image in raw_images:
            conditioning_text = ""
            inputs = processor(image, conditioning_text, return_tensors="pt")
            out = model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)
            descriptions.append(description)
        
        return descriptions


    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    

if __name__ == "__main__":
    update_env_vars(".env")

    rag = RAG()

    # This is how you find out what's in a link
    links = ["https://heits.digital/articles/gpt3-overview"]
    query = "Is this a good article for someone that wants to start learning Machine Learning?"
    answer = rag.query_links(links, query)

    print(answer)

    # This is how you find out what's in a image
    image_paths = [str(path) for path in Path("data/images/").glob("*")]
    query = "Are there any pokemon cards? And if so, describe them to me and give me the name of the file."
    answer = rag.query_images(image_paths, query)

    print(answer)