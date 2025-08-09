import os
import joblib
import logging
from typing import List, Optional, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentRetriever:
    """
    Kelas untuk mengambil dokumen relevan dengan logika reranking yang dapat dikonfigurasi.
    """

    def __init__(self, data_path: str = "data/perda_data.pkl"):
        self.data_path = data_path
        self.chunks: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self._load_data()

        try:
            # Tetap muat model reranker, penggunaannya akan bersifat opsional
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
            logging.info("Model CrossEncoder (reranker) berhasil dimuat.")
        except Exception as e:
            logging.error(f"Gagal memuat model CrossEncoder: {e}")
            self.reranker = None

    def __str__(self) -> str:
        is_reranker_loaded = "Yes" if self.reranker else "No"
        return f"<DocumentRetriever | chunks: {len(self.chunks)} | Reranker Loaded: {is_reranker_loaded}>"

    def _load_data(self):
        # ... (fungsi ini tidak berubah, sama seperti sebelumnya) ...
        if not os.path.exists(self.data_path):
            logging.error(f"File data tidak ditemukan: {self.data_path}.")
            return
        
        try:
            data = joblib.load(self.data_path)
            self.chunks = data.get('chunks', [])
            self.vectorizer = data.get('vectorizer')
            self.tfidf_matrix = data.get('tfidf_matrix')
            
            if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
                logging.error("Data yang dimuat tidak lengkap.")
                self.chunks, self.vectorizer, self.tfidf_matrix = [], None, None
                return

            logging.info(f"Data retriever (TF-IDF) berhasil dimuat. Total chunks: {len(self.chunks)}")
        except Exception as e:
            logging.error(f"Gagal memuat data dari {self.data_path}: {e}")
            self.chunks, self.vectorizer, self.tfidf_matrix = [], None, None

    # --- PERUBAHAN UTAMA DI SINI ---
    def retrieve_chunks(self, query: str, top_k: int = 5, initial_k: int = 50, use_reranker: bool = True) -> List[Tuple[str, float]]:
        """
        Mengambil potongan dokumen (chunks) yang relevan.
        
        Args:
            query (str): Pertanyaan pengguna.
            top_k (int): Jumlah hasil akhir yang diinginkan.
            initial_k (int): Jumlah kandidat awal yang diambil oleh TF-IDF (hanya digunakan jika reranker aktif).
            use_reranker (bool): Jika True, gunakan reranker. Jika False, kembalikan hasil TF-IDF.

        Returns:
            List[Tuple[str, float]]: Daftar tuple berisi (chunk, skor). Skor adalah dari reranker atau TF-IDF.
        """
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            logging.warning("Retriever TF-IDF tidak siap.")
            return []
            
        if not query.strip():
            return []
        
        # --- Tahap 1: Initial Retrieval (TF-IDF) ---
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Tentukan berapa banyak kandidat yang perlu diambil
        # Jika tidak pakai reranker, cukup ambil top_k. Jika pakai, ambil initial_k.
        num_candidates = initial_k if use_reranker and self.reranker else top_k
        
        # Ambil indeks kandidat teratas
        top_indices = cosine_similarities.argsort()[-num_candidates:][::-1]
        
        # --- Logika Pemilihan Versi ---
        
        # Versi 1: TANPA RERANKER (Baseline)
        if not use_reranker or not self.reranker:
            if not use_reranker:
                logging.info(f"Reranker tidak digunakan. Mengembalikan top {top_k} hasil dari TF-IDF.")
            else: # self.reranker is None but use_reranker was True
                logging.warning("Reranker diminta tetapi tidak tersedia. Mengembalikan hasil dari TF-IDF.")
            
            # Kembalikan hasil teratas dari TF-IDF beserta skornya
            results = [(self.chunks[i], cosine_similarities[i]) for i in top_indices if cosine_similarities[i] > 0]
            return results[:top_k]

        # Versi 2: DENGAN RERANKER
        initial_chunks = [self.chunks[i] for i in top_indices if cosine_similarities[i] > 0]
        if not initial_chunks:
            return []
            
        logging.info(f"TF-IDF menemukan {len(initial_chunks)} kandidat awal. Melanjutkan ke reranking...")
        
        rerank_pairs = [[query, chunk] for chunk in initial_chunks]
        scores = self.reranker.predict(rerank_pairs)
        
        scored_chunks = list(zip(initial_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        final_results = scored_chunks[:top_k]
        logging.info(f"Reranker selesai. Mengembalikan top {len(final_results)} hasil dengan skor.")
        
        return final_results