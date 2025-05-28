# %%
import base64
import concurrent.futures
import hashlib
import logging
import os
import shutil
import uuid
from io import BytesIO

import numpy as np
import torch
from colpali_engine.models import (
    ColPali,
    ColPaliProcessor,
    ColQwen2_5,
    ColQwen2_5_Processor,
)
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from dotenv import find_dotenv, load_dotenv
from openai import OpenAI
from pdf2image import convert_from_path
from PIL import Image
from pymilvus import DataType, MilvusClient
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from consts import PROJECT_ROOT_DIR, PROMPT

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# %%
model_name = "vidore/colqwen2.5-v0.2"
# model_name = "vidore/colpali-v1.2"
device = get_torch_device("cuda")

model = ColQwen2_5.from_pretrained(
    # model = ColPali.from_pretrained(
    pretrained_model_name_or_path=model_name,
    torch_dtype=torch.bfloat16,
    device_map=device,
).eval()

processor = ColQwen2_5_Processor.from_pretrained(
    # processor = ColPaliProcessor.from_pretrained(
    pretrained_model_name_or_path=model_name,
    use_fast=True,
)
_ = load_dotenv(find_dotenv(raise_error_if_not_found=True))
openai_client = OpenAI()


# %%
class MilvusManager:
    def __init__(self, milvus_uri, collection_name, create_collection, dim=128):
        """
        Initializes the MilvusManager.

        Args:
            milvus_uri (str): URI for Milvus server.
            collection_name (str): Name of the collection.
            create_collection (bool): Whether to create a new collection.
            dim (int, optional): Dimension of the vector. Defaults to 128.
        """
        self.client = MilvusClient(uri=milvus_uri)
        self.collection_name = collection_name
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.load_collection(collection_name)
        self.dim = dim
        self.max_doc_id = 0

        if create_collection:
            self.create_collection()
            self.create_index()

    def create_collection(self):
        """
        Creates a new collection in Milvus. Drops existing collection if present.
        """
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )
        schema.add_field(field_name="pk", datatype=DataType.INT64, is_primary=True)
        schema.add_field(
            field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dim
        )
        schema.add_field(field_name="seq_id", datatype=DataType.INT16)
        schema.add_field(field_name="doc_id", datatype=DataType.INT64)
        schema.add_field(field_name="doc", datatype=DataType.VARCHAR, max_length=65535)

        self.client.create_collection(
            collection_name=self.collection_name, schema=schema
        )

    def create_index(self):
        """
        Creates a vector index for the collection in Milvus.
        """
        self.client.release_collection(collection_name=self.collection_name)
        self.client.drop_index(
            collection_name=self.collection_name, index_name="vector"
        )
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_name="vector_index",
            index_type="FLAT",
            metric_type="IP",
            params={
                "M": 16,
                "efConstruction": 500,
            },
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def create_scalar_index(self):
        """
        Creates a scalar index for the doc_id field in Milvus.
        """
        self.client.release_collection(collection_name=self.collection_name)

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="doc_id",
            index_name="int32_index",
            index_type="INVERTED",
        )

        self.client.create_index(
            collection_name=self.collection_name, index_params=index_params, sync=True
        )

    def search(self, data, topk):
        """
        Searches for the top-k most similar documents in Milvus.

        Args:
            data (np.ndarray): Query vector.
            topk (int): Number of top results to return.

        Returns:
            list: List of (score, doc_id) tuples.
        """
        search_params = {"metric_type": "IP", "params": {}}
        results = self.client.search(
            self.collection_name,
            data,
            limit=50,
            output_fields=["vector", "seq_id", "doc_id"],
            search_params=search_params,
        )
        doc_ids = set()
        for r_id in range(len(results)):
            for r in range(len(results[r_id])):
                doc_ids.add(results[r_id][r]["entity"]["doc_id"])

        scores = []

        def rerank_single_doc(doc_id, data, client, collection_name):
            doc_colqwen_vecs = client.query(
                collection_name=collection_name,
                filter=f"doc_id in [{doc_id}, {doc_id + 1}]",
                output_fields=["seq_id", "vector", "doc"],
                limit=1000,
            )
            doc_vecs = np.vstack(
                [doc_colqwen_vecs[i]["vector"] for i in range(len(doc_colqwen_vecs))]
            )
            score = np.dot(data, doc_vecs.T).max(1).sum()
            return (score, doc_id)

        with concurrent.futures.ThreadPoolExecutor(max_workers=300) as executor:
            futures = {
                executor.submit(
                    rerank_single_doc, doc_id, data, self.client, self.collection_name
                ): doc_id
                for doc_id in doc_ids
            }
            for future in concurrent.futures.as_completed(futures):
                score, doc_id = future.result()
                scores.append((score, doc_id))

        scores.sort(key=lambda x: x[0], reverse=True)
        if len(scores) >= topk:
            return scores[:topk]
        else:
            return scores

    def insert(self, data):
        """
        Inserts a document's vectors and metadata into Milvus.

        Args:
            data (dict): Dictionary containing 'colqwen_vecs', 'doc_id', and 'filepath'.
        """
        colqwen_vecs = [vec for vec in data["colqwen_vecs"]]
        seq_length = len(colqwen_vecs)
        doc_ids = [data["doc_id"] for _ in range(seq_length)]
        seq_ids = list(range(seq_length))
        docs = [data["filepath"] for _ in range(seq_length)]
        # docs = [""] * seq_length
        # docs[0] = data["filepath"]

        self.client.insert(
            self.collection_name,
            [
                {
                    "vector": colqwen_vecs[i],
                    "seq_id": seq_ids[i],
                    "doc_id": doc_ids[i],
                    "doc": docs[i],
                }
                for i in range(seq_length)
            ],
        )

    def get_images_as_doc(self, images_with_vectors: list):
        """
        Converts a list of image vectors and filepaths into Milvus insertable format.

        Args:
            images_with_vectors (list): List of dicts with 'colqwen_vecs' and 'filepath'.

        Returns:
            list: List of dicts ready for Milvus insertion.
        """
        images_data = []

        for i in range(len(images_with_vectors)):
            self.max_doc_id += 1
            data = {
                "colqwen_vecs": images_with_vectors[i]["colqwen_vecs"],
                "doc_id": self.max_doc_id,
                "filepath": images_with_vectors[i]["filepath"],
            }
            images_data.append(data)

        return images_data

    def insert_images_data(self, image_data):
        """
        Inserts multiple images' data into Milvus.

        Args:
            image_data (list): List of image data dicts.
        """
        data = self.get_images_as_doc(image_data)

        for i in range(len(data)):
            self.insert(data[i])


# %%
class VectorProcessor:
    def __init__(
        self,
        id: str,
        create_collection=True,
    ):
        """
        Initializes the VectorProcessor with Milvus, Colqwen, and PDF managers.

        Args:
            id (str): Unique identifier for the session/user.
            create_collection (bool, optional): Whether to create a new collection. Defaults to True.
        """
        # hashed_id = hashlib.md5(id.encode()).hexdigest()[:8]
        # milvus_db_name = f"milvus_{hashed_id}.db"
        milvus_db_name = f"milvus_{id}.db"
        self.milvus_manager = MilvusManager(
            milvus_db_name, f"{id}", create_collection
        )
        self.colqwen_manager = ColqwenManager()
        self.pdf_manager = PdfManager()

    def index(
        self,
        pdf_path: str,
        id: str,
        max_pages: int,
    ):
        """
        Indexes a PDF file by converting pages to images, embedding them, and storing in Milvus.

        Args:
            pdf_path (str): Path to the PDF file.
            id (str): Unique identifier.
            max_pages (int): Maximum number of pages to process.

        Returns:
            list: List of saved image paths.
        """
        logger.info(f"Indexing {pdf_path}, id: {id}, max_pages: {max_pages}")

        image_paths = self.pdf_manager.save_images(id, pdf_path, max_pages)

        logger.info(f"Saved {len(image_paths)} images")

        colqwen_vecs = self.colqwen_manager.process_images(image_paths)

        images_data = [
            {"colqwen_vecs": colqwen_vecs[i], "filepath": image_paths[i]}
            for i in range(len(image_paths))
        ]

        logger.info(f"Inserting {len(images_data)} images data to Milvus")

        self.milvus_manager.insert_images_data(images_data)

        logger.info("Indexing completed")

        return image_paths

    def search(self, search_queries: list[str]):
        logger.info(f"Searching for {len(search_queries)} queries")

        final_res = []

        for query in search_queries:
            logger.info(f"Searching for query: {query}")
            query_vec = self.colqwen_manager.process_text([query])[0]
            search_res = self.milvus_manager.search(query_vec, topk=4)
            logger.info(f"Search result: {search_res} for query: {query}")
            final_res.append(search_res)

        return final_res


# %%
class PdfManager:
    def __init__(self):
        """
        Initializes the PdfManager.
        """
        pass

    def clear_and_recreate_dir(self, output_folder):
        logger.info(f"Clearing output folder {output_folder}")

        if os.path.exists(output_folder):
            shutil.rmtree(output_folder)

        os.makedirs(output_folder)

    def save_images(
        self, id, pdf_path, max_pages, pages: list[int] = None, output_folder=None
    ) -> list[str]:
        """
        Saves images of PDF pages to disk.

        Args:
            id (str): Unique identifier.
            pdf_path (str): Path to the PDF file.
            max_pages (int): Maximum number of pages to save.
            pages (list[int], optional): Specific pages to save. Defaults to None.

        Returns:
            list[str]: List of saved image file paths.
        """
        output_folder = (
            Path(output_folder) if output_folder is not None else output_folder
        )
        if output_folder is None:
            output_folder = Path(f"pages/{id}/")
        if not Path(output_folder).exists():
            Path(output_folder).mkdir(parents=True, exist_ok=True)
        images = convert_from_path(pdf_path=pdf_path)

        logger.info(
            f"Saving images from {pdf_path} to {output_folder}. Max pages: {max_pages}"
        )

        # self.clear_and_recreate_dir(output_folder)

        num_page_processed = 0

        for i, image in enumerate(images):
            if max_pages and num_page_processed >= max_pages:
                break

            if pages and i not in pages:
                continue

            full_save_path = output_folder / f"{id}_page_{i + 1}.png"

            # logger.debug(f"Saving image to {full_save_path}")

            image.save(fp=full_save_path, format="PNG")

            num_page_processed += 1

        return [
            f"{output_folder}/{id}_page_{i + 1}.png" for i in range(num_page_processed)
        ]


# %%
class ColqwenManager:
    def get_images(self, paths: list[str]) -> list[Image.Image]:
        """
        Loads images from file paths.

        Args:
            paths (list[str]): List of image file paths.

        Returns:
            list[Image.Image]: List of PIL Image objects.
        """
        return [Image.open(path) for path in paths]

    def process_images(self, image_paths: list[str], batch_size=5):
        logger.info(f"Processing {len(image_paths)} image_paths")

        images = self.get_images(image_paths)

        dataloader = DataLoader(
            dataset=ListDataset[str](images),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda x: processor.process_images(x),
        )

        ds: list[torch.Tensor] = []
        for batch_doc in tqdm(dataloader):
            with torch.no_grad():
                batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
                embeddings_doc = model(**batch_doc)
            ds.extend(list(torch.unbind(embeddings_doc.to(device))))

        ds_np = [d.float().cpu().numpy() for d in ds]

        return ds_np

    def process_text(self, texts: list[str]):
        logger.info(f"Processing {len(texts)} texts")

        dataloader = DataLoader(
            dataset=ListDataset[str](texts),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: processor.process_queries(x),
        )

        qs: list[torch.Tensor] = []
        for batch_query in dataloader:
            with torch.no_grad():
                batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
                embeddings_query = model(**batch_query)

            qs.extend(list(torch.unbind(embeddings_query.to(device))))

        qs_np = [q.float().cpu().numpy() for q in qs]

        return qs_np


# %%
# def generate_uuid(state):
#     """
#     Generates or retrieves a UUID for the user session.

#     Args:
#         state (dict): State dictionary containing 'user_uuid'.

#     Returns:
#         str: UUID string.
#     """
#     # Check if UUID already exists in session state
#     if state["user_uuid"] is None:
#         # Generate a new UUID if not already set
#         state["user_uuid"] = str(uuid.uuid4())

#     return state["user_uuid"]


class RAG:
    def __init__(self):
        """
        Initializes the RAG.
        """
        self.vectordb_id = None
        self.img_path_dir = PROJECT_ROOT_DIR / "src/pages/"
    def create_vector_db(self, vectordb_id="policy_wordings", dir=PROJECT_ROOT_DIR/"data" , max_pages=200):
        """
        Uploads a PDF file, converts it to images, and indexes it.

        Args:
            state (dict): State dictionary for user session.
            file: Uploaded file object.
            max_pages (int, optional): Maximum number of pages to process. Defaults to 100.

        Returns:
            str: Status message.
        """

        logger.info(f"Converting files in: {dir}.")

        try:
            for idx,f in enumerate((dir/"policy_wordings").iterdir()):
                if idx==0:
                    vectorprocessor = VectorProcessor(id=vectordb_id, create_collection=True)
                    self.vectordb_id = vectordb_id
                _ = vectorprocessor.index(pdf_path=f, id=f.stem, max_pages=max_pages)
            return f"✅ Created the vector_db: milvus_{vectordb_id} under `src` dir."
        except Exception as err:
            return f"❌ Error creating vector_db: {err}"

    def search_documents(self, query):
        if self.vectordb_id is None:
            raise Exception("Create the vector db first by invoking `create_vector_db`.")
        try:
            vectorprocessor = VectorProcessor(id=self.vectordb_id, create_collection=False)

            search_results = vectorprocessor.search(search_queries=[query])[0]

            check_res = vectorprocessor.milvus_manager.client.query(collection_name=self.vectordb_id,
                                       filter=f"doc_id in {[d[1] for d in search_results]}",
                                       output_fields=[ "doc_id", "doc"])
            img_path_doc_id = set((i['doc'], i['doc_id']) for i in check_res)

            logger.info("✅ Retrieved the images for answering query.")
            return img_path_doc_id

        except Exception as err:
            return f"❌ Error during search: {err}"

    def encode_image_to_base64(self, image_path):
        """
        Encodes an image file to a base64 string.

        Args:
            image_path (str): Path to the image file.

        Returns:
            str: Base64 encoded image string.
        """
        """Encodes a PIL image to a base64 string."""
        image = Image.open(image_path)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def query_gpt4o_mini(self, query, image_path):
        """
        Queries the OpenAI GPT-4o-mini model with a query and images.

        Args:
            query (str): The user query.
            image_path (list): List of image file paths.

        Returns:
            str: The AI response.
        """
        try:
            base64_images = [self.encode_image_to_base64(pth) for pth in image_path]

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": PROMPT.format(query=query)}
                        ]
                        + [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{im}"},
                            }
                            for im in base64_images
                        ],
                    }
                ],
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as err:
            return f"Unable to generate the final output due to: {err}."


# %%
# def main():
#     """
#     Main function for running the PDF search app in standalone mode.
#     """
#     user_id = {"user_uuid": None}
#     app = RAG()
#     app.upload_and_convert(
#         state=user_id, file=PROJECT_ROOT_DIR / "data/PI-TOOL-WEB-SAMPLE.pdf"
#     )
#     logger.info(user_id)
#     _, response = app.search_documents(
#         state=user_id, query="what's JPGCCOMP allocation?"
#     )
#     logger.info(response)


# %%
# if __name__ == "__main__":
#     main()

# %%
# app = RAG()
# %%
# for idx,f in enumerate((PROJECT_ROOT_DIR/"data/policy_wordings").iterdir()):
#     if idx==0:
#         vectorprocessor = VectorProcessor(id="policy_wordings", create_collection=True)
#     pages = vectorprocessor.index(pdf_path=f, id=f.stem, max_pages=200)
# %%
vectorprocessor = VectorProcessor(id="policy_wordings", create_collection=False)
# %%
query = "what critical illnesses are covered under optima restore?"
query_vec = vectorprocessor.colqwen_manager.process_text([query])[0]
# retrive only 4 as only 4 images can be inferenced at a time with QWEN 2.5 72B
search_res = vectorprocessor.milvus_manager.search(query_vec, topk=4) 
# %%
search_res
# %%
vectorprocessor.search([query])
# %%
check_res = vectorprocessor.milvus_manager.client.query(collection_name="policy_wordings",
                                       filter=f"doc_id in {[d[1] for d in search_res]}",
                                       output_fields=[ "doc_id", "doc"])
# %%
set((i['doc'], i['doc_id']) for i in check_res), search_res
# %%
search_params = {"metric_type": "IP", "params": {}}
results = vectorprocessor.milvus_manager.client.search(
    collection_name="policy_wordings",
    data=query_vec,
    limit=2,
    output_fields=["vector", "seq_id", "doc_id", "doc"],
    search_params=search_params,
)
# %%
results[0][1]
# %%
a = vectorprocessor.milvus_manager.client.query(
    collection_name="policy_wordings", filter="doc_id == 45", output_fields=["doc"]
)

# %%
a[3]
# %%
vectorprocessor.milvus_manager.client.query(
    collection_name="policy_wordings",
    filter="pk == 458295805050210631",
    output_fields=["doc", "doc_id"],
)
# %%
# We have to change the search to search by pk and not doc_id

# %%
rag_app = RAG()
rag_app.vectordb_id = "policy_wordings"

# %%
rag_app.search_documents(query)
# %%
