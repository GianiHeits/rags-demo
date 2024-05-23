import os
from pathlib import Path
from dotenv import dotenv_values

from langchain import hub
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.output_parsers import PydanticOutputParser

from transformers import BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering

from src.youtube_source import get_transcript_from_url

from pydantic import BaseModel
from typing import List
from PIL import Image


class Questions(BaseModel):
    questions: List[str]


def update_env_vars(env_file_path: str=None):
    if env_file_path:
        env_config = dotenv_values(env_file_path)
        os.environ = {**os.environ, **env_config}

class RAG:
    def __init__(self, streaming=False):
        self.encoder = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo-0125", streaming=streaming)

        self.blip_model_type = "Salesforce/blip-image-captioning-large"
        self.blip_processor = BlipProcessor.from_pretrained(self.blip_model_type)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(self.blip_model_type)
        
        self.blip_qna_model_type = "Salesforce/blip-vqa-base"
        self.blip_qna_processor = BlipProcessor.from_pretrained(self.blip_qna_model_type)
        self.blip_qna_model = BlipForQuestionAnswering.from_pretrained(self.blip_qna_model_type)

        self.vectorstore_links_sources = None
        self.vectorstore_links = None
        
        self.vectorstore_youtube_sources = None
        self.vectorstore_youtube = None

        self.vectorstore_images_sources = None
        self.vectorstore_images = None


    def prepare_vectorstores_for_demo(self):
        self.vectorstore_links_sources = ["https://heits.digital/articles/gpt3-overview"]
        self.vectorstore_links = self._get_vectorstore_from_links(self.vectorstore_links_sources)

        self.vectorstore_youtube_sources = "https://www.youtube.com/watch?v=lW7Mxj8KUJE&ab_channel=LinusTechTips"
        self.vectorstore_youtube = self._get_vectorstore_from_youtube(self.vectorstore_youtube_sources)

        self.vectorstore_images_sources = [str(path) for path in Path("data/images/").glob("*")]
        raw_images = [Image.open(img_path).convert('RGB') for img_path in self.vectorstore_images_sources]
        self.vectorstore_images = self._get_extensive_vectorstore_from_images(raw_images, self.vectorstore_images_sources)


    def query_links(self, links: List[str], query: str) -> str:
        if links != self.vectorstore_links_sources:
            self.vectorstore_links_sources = links
            self.vectorstore_links = self._get_vectorstore_from_links(links)

        rag_chain = self.get_rag_chain(self.vectorstore_links)
        return rag_chain.invoke(query)


    def query_images(self, image_path_list: List[str], query: str) -> str:
        if image_path_list != self.vectorstore_images_sources:
            raw_images = [Image.open(img_path).convert('RGB') for img_path in image_path_list]
            self.vectorstore_images_sources = raw_images
            self.vectorstore_images = self._get_extensive_vectorstore_from_images(raw_images, image_path_list)

        rag_chain = self.get_rag_chain(self.vectorstore_images)
        return rag_chain.invoke(query)


    def query_youtube_video(self, video_url: str, query: str) -> str:
        if video_url != self.vectorstore_youtube_sources:
            self.vectorstore_youtube_sources = video_url
            self.vectorstore_youtube = self._get_vectorstore_from_youtube(video_url)

        rag_chain = self.get_rag_chain(self.vectorstore_youtube, 15)
        return rag_chain.invoke(query)


    def get_rag_chain(self, vectorstore, top_k=5):
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": top_k})
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
    

    def _get_vectorstore_from_images(self, raw_images_list: List[Image.Image], raw_images_paths: List[str]=None, query: str=""):
        # Use this to ask GPT about content in images
        descriptions = self._extract_generic_image_info(raw_images_list)
        descriptions_questions = self._extract_image_info_based_on_questions(raw_images_list, question=query)

        documents = []
        for i, [desc1, desc2] in enumerate(zip(descriptions, descriptions_questions)):
            source = raw_images_paths[i] if raw_images_paths != None else f"image_{i}"
            description = "{" + f"file_name: '{source}', description_generic: '{desc1}', description_specific: '{desc2}'" + "}"
            doc = Document(page_content=description, metadata={"source": source})
            documents.append(doc)
        print(documents)
        
        vectorstore = Chroma.from_documents(documents=documents, embedding=self.encoder)
        return vectorstore


    def _get_extensive_vectorstore_from_images(self, raw_images_list: List[Image.Image], raw_images_paths: List[str]=None):
        descriptions = self._extract_extensive_image_info(raw_images_list)

        documents = []
        for i, desc in enumerate(descriptions):
            source = raw_images_paths[i] if raw_images_paths != None else f"image_{i}"
            description = "{" + f"file_name: '{source}', description: '{desc}'" + "}"
            doc = Document(page_content=description, metadata={"source": source})
            documents.append(doc)
        
        vectorstore = Chroma.from_documents(documents=documents, embedding=self.encoder)
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
        
        vectorstore = Chroma.from_documents(documents=documents, embedding=self.encoder)
        return vectorstore

    def _get_questions_from_prompt(self, query):
        parser = PydanticOutputParser(pydantic_object=Questions)
        formatting = parser.get_format_instructions()

        prompt = ChatPromptTemplate.from_template(
            "Pretend you are Sherlock Holmes and you ask as many questions in order to solve a mistery. What questions would ask about a image with the following description: {query}. {format_instructions}", 
            partial_variables={"format_instructions": formatting}
            )
        chain = prompt | self.llm | JsonOutputParser()
        return chain.invoke(query)

    def _rephrase_question_answer(self, question, answer):
        prompt = ChatPromptTemplate.from_template(
            "Rephrase the provided question based on the answer in order to be a statement instead of question. print out just the statement itself, no other text. \nquestion: {question}\nanswer: {answer}"
        )
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"question": question, "answer": answer})

    def _extract_generic_image_info(self, raw_images: List):
        descriptions = []

        # conditional image captioning
        for image in raw_images:
            conditioning_text = ""
            inputs = self.blip_processor(image, conditioning_text, return_tensors="pt")
            out = self.blip_model.generate(**inputs, max_new_tokens=20)
            description = self.blip_processor.decode(out[0], skip_special_tokens=True)
            print(description)
            descriptions.append(description)
        
        return descriptions

    def _extract_image_info_based_on_questions(self, raw_images: List, question=""):
        descriptions = []

        # conditional image captioning
        for image in raw_images:
            inputs = self.blip_qna_processor(image, question, return_tensors="pt")
            out = self.blip_qna_model.generate(**inputs, max_new_tokens=20)
            description = self.blip_qna_processor.decode(out[0], skip_special_tokens=True)
            print(description)
            descriptions.append(description)
        
        return descriptions
    
    def _extract_extensive_image_info(self, raw_images: List):
        # get basic description of the images
        base_descriptions = self._extract_generic_image_info(raw_images)
        
        extensive_descriptions = []
        for image, description in zip(raw_images, base_descriptions):
            # generate questions based on the basic description
            questions = self._get_questions_from_prompt(description)
            print('Questions: ', questions["questions"])

            # get a more extensive description based on questions
            image_descriptions = []
            for question in questions["questions"]:
                print("Asking: ", question)
                inputs = self.blip_qna_processor(image, question, return_tensors="pt")
                out = self.blip_qna_model.generate(**inputs, max_new_tokens=20)
                
                answer = self.blip_qna_processor.decode(out[0], skip_special_tokens=True)
                print("Answer: ", answer)

                rephrased = self._rephrase_question_answer(question, answer)
                rephrased = rephrased.replace("Statement:", "")
                print("Rephrased: ", rephrased)

                image_descriptions.append(rephrased)
            
            extensive_descriptions.append(" ".join(image_descriptions))
        return extensive_descriptions

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

    # video_url = "https://www.youtube.com/watch?v=lW7Mxj8KUJE&ab_channel=LinusTechTips"
    # query = "Why is google chrome slow based on the transcript?"
    # answer = rag.query_youtube_video(video_url, query)
    
    # video_url = "https://www.youtube.com/watch?v=JDEc9Z_LI9I&ab_channel=SomeOrdinaryGamers"
    # query = "Why will the speaker won't buy EA games anymore?"
    # answer = rag.query_youtube_video(video_url, query)

    # This is how you find out what's in a image
    image_paths = [str(path) for path in Path("data/images/").glob("*")]
    raw_images = [Image.open(img_path).convert('RGB') for img_path in image_paths]
    # query = "Are there any pokemon cards? And if so, what colors are the pokemons?."
    # answer = rag.query_images(image_paths, query)
    descriptions = rag._extract_extensive_image_info(raw_images)

    print(descriptions)