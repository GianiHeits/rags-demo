import os
from pathlib import Path
from dotenv import dotenv_values

from langchain import hub
from langchain_community.llms import Ollama
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings

from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser

from src.youtube_source import get_transcript_from_url

from pydantic import BaseModel
from typing import List
from PIL import Image

llm = Ollama(model="llama2")


class Questions(BaseModel):
    questions: List[str]


def update_env_vars(env_file_path: str = None):
    if env_file_path:
        env_config = dotenv_values(env_file_path)
        os.environ = {**os.environ, **env_config}


class RAG:
    def __init__(self):
        # OpenAI
        # self.encoder = OpenAIEmbeddings()
        # self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

        # Ollama
        self.encoder = OllamaEmbeddings()
        self.llm = Ollama(model="llama2")

    def query_links(self, links: List[str], query: str) -> str:
        vectorstore = self._get_vectorstore_from_links(links)
        rag_chain = self.get_rag_chain(vectorstore)
        answer = rag_chain.invoke(query)
        vectorstore.delete_collection()
        return answer

    def query_images(self, image_path_list: List[str], query: str) -> str:
        raw_images = [Image.open(img_path).convert('RGB')
                      for img_path in image_path_list]
        vectorstore = self._get_vectorstore_from_images(
            raw_images, image_path_list, query)
        rag_chain = self.get_rag_chain(vectorstore)
        answer = rag_chain.invoke(query)
        vectorstore.delete_collection()
        return answer

    def query_youtube_video(self, video_url: str, query: str) -> str:
        vectorstore = self._get_vectorstore_from_youtube(video_url)
        rag_chain = self.get_rag_chain(vectorstore, 15)
        answer = rag_chain.invoke(query)
        vectorstore.delete_collection()
        return answer

    def get_rag_chain(self, vectorstore, top_k=5):
        retriever = vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k})
        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = (
            {"context": retriever | self.format_docs,
                "question": RunnablePassthrough()}
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

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(
            documents=splits, embedding=self.encoder)

        return vectorstore

    def _get_vectorstore_from_images(self, raw_images_list: List[Image.Image], raw_images_paths: List[str] = None, query: str = ""):
        # Use this to ask GPT about content in images
        descriptions = self._extract_generic_image_info(raw_images_list)
        # descriptions_questions = self._extract_image_info_based_on_questions(raw_images_list, question=query)

        documents = []
        for i, desc in enumerate(descriptions):
            source = raw_images_paths[i] if raw_images_paths != None else f"image_{i}"
            pc = "{" + \
                f"file_name: '{source}', description: '{desc}'" + "}"
            doc = Document(page_content=pc,
                           metadata={"source": source})
            documents.append(doc)
        print(documents)

        vectorstore = Chroma.from_documents(
            documents=documents, embedding=self.encoder)
        return vectorstore

    def _get_vectorstore_from_youtube(self, video_url):
        # Use this to ask GPT about content in videos
        transcript = get_transcript_from_url(video_url)

        documents = []
        for item in transcript:
            metadata = item.copy()
            page_content = metadata.pop('text', None)
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)

        vectorstore = Chroma.from_documents(
            documents=documents, embedding=self.encoder)
        return vectorstore

    def _get_questions_from_prompt(self, query):
        # parser = PydanticOutputParser(pydantic_object=Questions)
        # formatting = parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_template(
            # "Pretend you are Sherlock Holmes and you only ask as few questions as possbile to unravel a mistery. What would standalone question would ask about a image to answer the following mistery: {query}. {format_instructions}",
            "You will receive a text description of an image. Based on this, provide 3 follow-up questions that can't be answered with yes or no directly, that would allow me to better understand what's in the image. output everything as a simple 1,2,3 list, no other text. description: {query}",
            # partial_variables={"format_instructions": formatting}
        )
        chain = prompt | self.llm
        return chain.invoke(query)

    def _rephrase_question_answer(self, question, answer):
        prompt = ChatPromptTemplate.from_template(
            "Rephrase the provided question based on the answer in order to be a statement instead of question. print out just the statement itself, no other text. \nquestion: {query}\nanswer: {answer}",
            partial_variables={"answer": answer}
        )
        chain = prompt | self.llm
        return chain.invoke(question)

    def _extract_generic_image_info(self, raw_images: List, model_type="Salesforce/blip-image-captioning-large"):
        from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

        descriptions = []

        processor = BlipProcessor.from_pretrained(model_type)
        model = BlipForConditionalGeneration.from_pretrained(model_type)
        model_questioning = BlipForQuestionAnswering.from_pretrained(
            "Salesforce/blip-vqa-base")

        # conditional image captioning
        for image in raw_images:
            image_descs = []
            conditioning_text = ""
            inputs = processor(image, conditioning_text, return_tensors="pt")
            out = model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)
            print("Image description: ")
            print(description)

            # let's get some more specific questions
            questions = self._get_questions_from_prompt(description)

            questions_list = questions.split("\n")
            print("Questions list: ")
            print(questions_list)
            replies_list = []
            # let's go thru the questions and get replies from model
            for desc in questions_list:
                if len(desc) > 1 and (desc[0] == '1' or desc[0] == '2' or desc[0] == '3'):
                    print("Asking: ")
                    print(desc)
                    inputs = processor(image, desc, return_tensors="pt")
                    out = model_questioning.generate(**inputs)
                    answer = processor.decode(out[0], skip_special_tokens=True)
                    print("Answer: ")
                    print(answer)

                    # rephrase
                    rephrased = self._rephrase_question_answer(desc, answer)
                    rephrased = rephrased.replace("Statement:", "")
                    print("rephrased: ")
                    print(rephrased)

                    replies_list.append(rephrased)

            image_descs.append(description)
            image_descs.extend(replies_list)

            descriptions.append("\n".join(image_descs))

        return descriptions

    def _extract_image_info_based_on_questions(self, raw_images: List, question="", model_type="Salesforce/blip-vqa-base"):
        from transformers import BlipProcessor, BlipForQuestionAnswering

        descriptions = []

        processor = BlipProcessor.from_pretrained(model_type)
        model = BlipForQuestionAnswering.from_pretrained(model_type)

        # conditional image captioning
        for image in raw_images:
            inputs = processor(image, question, return_tensors="pt")
            out = model.generate(**inputs)
            description = processor.decode(out[0], skip_special_tokens=True)
            print(description)
            descriptions.append(description)

        return descriptions

    @staticmethod
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    update_env_vars(".env")

    rag = RAG()

    # # This is how you find out what's in a link
    # links = ["https://rust-exercises.com/"]
    # query = "Is this a good article for someone that wants to start learning Machine Learning?"
    # answer = rag.query_links(links, query)

    # print(answer)

    # video_url = "https://www.youtube.com/watch?v=lW7Mxj8KUJE&ab_channel=LinusTechTips"
    # query = "Why is google chrome slow based on the transcript?"
    # answer = rag.query_youtube_video(video_url, query)

    # video_url = "https://www.youtube.com/watch?v=JDEc9Z_LI9I&ab_channel=SomeOrdinaryGamers"
    # query = "Why will the speaker won't buy EA games anymore?"
    # answer = rag.query_youtube_video(video_url, query)

    # print(answer)

    # This is how you find out what's in a image
    image_paths = [str(path) for path in Path("data/images/").glob("*")]
    query = "Are there any pokemon cards? And if so, what colors are the pokemons?."
    answer = rag.query_images(image_paths, query)

    print(answer)
