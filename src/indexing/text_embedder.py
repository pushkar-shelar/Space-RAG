from sentence_transformers import SentenceTransformer


class TextEmbedder:

    def __init__(
        self,
        model_name="all-MiniLM-L6-v2"
    ):

        print("Loading text embedding model...")

        self.model = SentenceTransformer(
            model_name
        )

        print("Text embedding model loaded.")


    def embed_texts(self, texts):

        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True
        )

        return embeddings


    def embed_query(self, query):

        embedding = self.model.encode(
            query,
            convert_to_tensor=True
        )

        return embedding