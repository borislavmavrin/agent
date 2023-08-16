import faiss
import torch
from transformers import AutoTokenizer, AutoModel
import re
import spacy


alice = '''In another moment down went Alice after it, never once considering how
in the world she was to get out again.
The rabbit-hole went straight on like a tunnel for some way, and then
dipped suddenly down, so suddenly that Alice had not a moment to think
about stopping herself before she found herself falling down what seemed
to be a very deep well. Dummy sentence 1. Dummy sentence 2.'''


class RetrieverConfig:
    indexing_dimension = 768
    index_factory_string = "Flat"
    metric = faiss.METRIC_INNER_PRODUCT
    embedder = "facebook/contriever"


class Retriever(object):

    def __init__(self, config, corpus):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.faiss_index = faiss.index_factory(config.indexing_dimension, config.index_factory_string,
                                               config.metric)
        self._tokenizer = None
        self._embedder = None
        self.nlp = spacy.load('en_core_web_sm')

        self.init_embedder(config.embedder)
        self.id_to_chunk = dict()
        self.last_idx = -1
        self.index_corpus(corpus)

    def init_embedder(self, model):
        self._tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        model = AutoModel.from_pretrained('facebook/contriever')
        model.to(self.device)
        self._embedder = model

    def embed(self, text):
        inputs = self._tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        inputs.to(self.device)
        outputs = self._embedder(**inputs)

        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings

        embeddings = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
        return embeddings.numpy()

    def add_to_index(self, chunk):
        chunk = chunk.strip()
        chunk_vec = self.embed(chunk)
        self.faiss_index.add(chunk_vec)
        idx = self.last_idx + 1
        self.id_to_chunk[idx] = chunk
        self.last_idx += 1

    def index_corpus(self, corpus):
        r = re.compile(r"^\s+", re.MULTILINE)
        corpus = r.sub(" ", corpus)
        corpus = corpus.replace("\n", " ")
        doc = self.nlp(corpus)
        for chunk in doc.sents:
            chunk = str(chunk)
            if len(chunk.split()) > 1:
                self.add_to_index(chunk)

    def search_knn(self, query, top_k):
        query_vector = self.embed(query)
        query_vector = query_vector.astype('float32')
        scores, indexes = self.faiss_index.search(query_vector, top_k)
        top_chunks = [self.id_to_chunk[idx] for idx in indexes[0] if idx >= 0]
        return top_chunks, scores


if __name__ == '__main__':
    retriever = Retriever(RetrieverConfig, alice)
    top_chunks, scores = retriever.search_knn("rabbit-hole", 1)
    assert top_chunks[0] == retriever.id_to_chunk[1]

