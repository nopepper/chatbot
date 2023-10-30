from haystack.document_stores import FAISSDocumentStore
from haystack.utils import convert_files_to_docs
from haystack.nodes import EmbeddingRetriever, BaseRetriever
from haystack.pipelines import ExtractiveQAPipeline
from os import path
import os
from typing import List
from haystack import Document
from haystack.document_stores import BaseDocumentStore
import glob


def get_docs(data_dir="./data/") -> List[Document]:
    from haystack import Pipeline
    from haystack.nodes import TextConverter, PreProcessor

    indexing_pipeline = Pipeline()
    text_converter = TextConverter()
    preprocessor = PreProcessor(
        clean_whitespace=True,
        clean_header_footer=True,
        clean_empty_lines=True,
        split_by="word",
        split_length=200,
        split_overlap=20,
        split_respect_sentence_boundary=True,
    )
    indexing_pipeline.add_node(component=text_converter, name="TextConverter", inputs=["File"])
    indexing_pipeline.add_node(component=preprocessor, name="PreProcessor", inputs=["TextConverter"])

    return indexing_pipeline.run_batch(file_paths=glob.glob(path.join(data_dir, "**/*.*"), recursive=True))['documents'] # type: ignore


def get_docs2(data_dir="./data/") -> List[Document]:
    return convert_files_to_docs(dir_path=data_dir, split_paragraphs=False)


def get_retriever(
        data_dir="./data/", 
        index_dir="embeddings/faiss",
        force_rebuild=False
    ) -> BaseRetriever:

    index_path = path.join(index_dir, "faiss.emb")
    index_config_path = path.join(index_dir, "faiss.json")

    if force_rebuild:
        if path.exists(index_path):
            os.remove(index_path)
        if path.exists(index_config_path):
            os.remove(index_config_path)
        if path.exists(path.join(index_dir, "faiss_document_store.db")):
            os.remove(path.join(index_dir, "faiss_document_store.db"))

    if not path.exists(index_path):
        document_store = FAISSDocumentStore(
            faiss_index_factory_str="Flat", 
            sql_url=f"sqlite:///{index_dir}/faiss_document_store.db"
        )
    else:
        document_store = FAISSDocumentStore.load(index_path)

    retriever = EmbeddingRetriever(
            document_store=document_store, 
            embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
            top_k=3
        )

    if not path.exists(index_path):
        print("Building index...")

        docs = get_docs(data_dir=data_dir)
        document_store.write_documents(docs)
        document_store.update_embeddings(retriever)
        document_store.save(index_path)

    return retriever