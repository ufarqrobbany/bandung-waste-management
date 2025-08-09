# import os
# import joblib
# import logging
# from typing import List, Optional
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# # Konfigurasi logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class DocumentRetriever:
#     """
#     Kelas untuk memuat data dokumen yang telah diproses dan mengambil bagian (chunk) 
#     yang paling relevan terhadap query menggunakan TF-IDF dan cosine similarity.
#     Ini adalah implementasi baseline tanpa semantic reranking.
#     """

#     def __init__(self, data_path: str = "data/perda_data.pkl"):
#         """
#         Inisialisasi DocumentRetriever.

#         Args:
#             data_path (str): Path ke file pickle yang berisi data dokumen, vectorizer, dan TF-IDF matrix.
#         """
#         self.data_path = data_path
#         self.chunks: List[str] = []
#         self.vectorizer: Optional[TfidfVectorizer] = None
#         self.tfidf_matrix: Optional[np.ndarray] = None
#         self._load_data()

#     def __str__(self) -> str:
#         """Representasi string dari objek."""
#         return f"<DocumentRetriever | chunks: {len(self.chunks)}>"

#     def _load_data(self):
#         """
#         Memuat data dari file pickle yang telah diproses sebelumnya.
#         File ini harus berisi kunci 'chunks', 'vectorizer', dan 'tfidf_matrix'.
#         """
#         if not os.path.exists(self.data_path):
#             logging.error(f"File data tidak ditemukan: {self.data_path}. Jalankan skrip pemrosesan data.")
#             return
        
#         try:
#             data = joblib.load(self.data_path)
#             self.chunks = data.get('chunks', [])
#             self.vectorizer = data.get('vectorizer')
#             self.tfidf_matrix = data.get('tfidf_matrix')
            
#             if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
#                 logging.error("Data yang dimuat tidak lengkap. Pastikan file data valid.")
#                 self.chunks = []
#                 self.vectorizer = None
#                 self.tfidf_matrix = None
#                 return

#             logging.info(f"Data retriever berhasil dimuat. Total chunks: {len(self.chunks)}")
#         except Exception as e:
#             logging.error(f"Gagal memuat data dari {self.data_path}: {e}")
#             self.chunks = []
#             self.vectorizer = None
#             self.tfidf_matrix = None

#     def retrieve_chunks(self, query: str, top_k: int = 10) -> List[str]:
#         """
#         Mengambil potongan dokumen (chunks) yang paling relevan terhadap query yang diberikan
#         menggunakan cosine similarity terhadap representasi TF-IDF.

#         Args:
#             query (str): Pertanyaan atau masukan dari pengguna.
#             top_k (int): Jumlah hasil paling relevan yang ingin dikembalikan.

#         Returns:
#             List[str]: Daftar chunks teks yang paling relevan terhadap query.
#         """
#         if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
#             logging.warning("Retriever tidak siap. Kembalikan array kosong.")
#             return []
            
#         if not query.strip():
#             logging.info("Query kosong. Tidak ada retrieval yang dilakukan.")
#             return []
        
#         try:
#             query_vector = self.vectorizer.transform([query])
#             cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
#             top_k_indices = cosine_similarities.argsort()[-top_k:][::-1]
            
#             relevant_chunks = [self.chunks[i] for i in top_k_indices]
            
#             logging.info(f"Ditemukan {len(relevant_chunks)} chunks relevan untuk query.")
#             return relevant_chunks
#         except Exception as e:
#             logging.error(f"Gagal melakukan retrieval: {e}")
#             return []

# # Contoh penggunaan
# if __name__ == "__main__":
#     # Catatan: File data/perda_data.pkl harus sudah dibuat menggunakan skrip pemrosesan data.
#     retriever = DocumentRetriever()
#     print(retriever)
    
#     test_query = "Bagaimana cara membuang sachet kopi?"
#     relevant_docs = retriever.retrieve_chunks(test_query, top_k=10)
    
#     print("\n--- Hasil Retrieval (TF-IDF Baseline) ---")
#     print(f"Query: {test_query}")
#     for i, chunk in enumerate(relevant_docs):
#         print(f"\nChunk {i+1}:\n{chunk[:250]}...")

import os
import joblib
import logging
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import CrossEncoder # <-- Tambahan baru

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentRetriever:
    """
    Kelas untuk mengambil dokumen relevan.
    Tahap 1: Pengambilan cepat dengan TF-IDF (Initial Retrieval).
    Tahap 2: Pemeringkatan ulang (reranking) dengan model semantik (Semantic Reranking).
    """

    def __init__(self, data_path: str = "data/perda_data.pkl"):
        """
        Inisialisasi DocumentRetriever. Memuat data TF-IDF dan model reranker.
        """
        self.data_path = data_path
        self.chunks: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self._load_data()

        # --- Langkah Tambahan: Muat model Reranker ---
        try:
            # Menggunakan model yang dioptimalkan untuk tugas semantic similarity
            self.reranker = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2')
            logging.info("Model CrossEncoder (reranker) berhasil dimuat.")
        except Exception as e:
            logging.error(f"Gagal memuat model CrossEncoder: {e}")
            self.reranker = None

    def __str__(self) -> str:
        """Representasi string dari objek."""
        is_reranker_loaded = "Yes" if self.reranker else "No"
        return f"<DocumentRetriever | chunks: {len(self.chunks)} | Reranker Loaded: {is_reranker_loaded}>"

    def _load_data(self):
        """
        Memuat data dari file pickle yang telah diproses sebelumnya.
        """
        if not os.path.exists(self.data_path):
            logging.error(f"File data tidak ditemukan: {self.data_path}. Jalankan skrip pemrosesan data.")
            return
        
        try:
            data = joblib.load(self.data_path)
            self.chunks = data.get('chunks', [])
            self.vectorizer = data.get('vectorizer')
            self.tfidf_matrix = data.get('tfidf_matrix')
            
            if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
                logging.error("Data yang dimuat tidak lengkap. Pastikan file data valid.")
                # Reset all to ensure consistent state
                self.chunks, self.vectorizer, self.tfidf_matrix = [], None, None
                return

            logging.info(f"Data retriever (TF-IDF) berhasil dimuat. Total chunks: {len(self.chunks)}")
        except Exception as e:
            logging.error(f"Gagal memuat data dari {self.data_path}: {e}")
            self.chunks, self.vectorizer, self.tfidf_matrix = [], None, None

    def retrieve_chunks(self, query: str, top_k: int = 10, initial_k: int = 100) -> List[str]:
        """
        Mengambil potongan dokumen (chunks) yang paling relevan.
        Proses: TF-IDF retrieval -> Semantic Reranking.

        Args:
            query (str): Pertanyaan dari pengguna.
            top_k (int): Jumlah hasil akhir yang paling relevan (setelah reranking).
            initial_k (int): Jumlah kandidat awal yang diambil oleh TF-IDF.

        Returns:
            List[str]: Daftar chunks teks yang paling relevan setelah di-rerank.
        """
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            logging.warning("Retriever TF-IDF tidak siap. Mengembalikan list kosong.")
            return []
            
        if not query.strip():
            logging.info("Query kosong. Tidak ada retrieval yang dilakukan.")
            return []
        
        # --- Tahap 1: Initial Retrieval (TF-IDF) ---
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Ambil 'initial_k' kandidat teratas
        top_initial_indices = cosine_similarities.argsort()[-initial_k:][::-1]
        initial_chunks = [self.chunks[i] for i in top_initial_indices if cosine_similarities[i] > 0]
        
        if not initial_chunks:
            logging.info("Tidak ada kandidat awal yang ditemukan oleh TF-IDF.")
            return []
        logging.info(f"TF-IDF menemukan {len(initial_chunks)} kandidat awal.")

        # --- Tahap 2: Semantic Reranking ---
        if not self.reranker:
            logging.warning("Reranker tidak tersedia. Mengembalikan hasil dari TF-IDF.")
            return initial_chunks[:top_k]
            
        # Buat pasangan [query, chunk] untuk di-score oleh reranker
        rerank_pairs = [[query, chunk] for chunk in initial_chunks]
        
        # Dapatkan skor relevansi semantik
        scores = self.reranker.predict(rerank_pairs)
        
        # Gabungkan chunks dengan skornya dan urutkan
        scored_chunks = list(zip(initial_chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Ambil chunks terbaik setelah diurutkan ulang
        reranked_chunks = [chunk for chunk, score in scored_chunks]
        
        logging.info(f"Reranker selesai memproses. Mengembalikan top {top_k} hasil.")
        return reranked_chunks[:top_k]

# Contoh penggunaan
if __name__ == "__main__":
    retriever = DocumentRetriever()
    print(retriever)
    
    test_query = "Bagaimana cara membuang sachet kopi?"
    # Hanya meminta 3 dokumen teratas setelah proses reranking
    relevant_docs = retriever.retrieve_chunks(test_query, top_k=3)
    
    print("\n" + "="*50)
    print("--- Hasil Retrieval (Setelah Reranking) ---")
    print(f"Query: {test_query}")
    if relevant_docs:
        for i, chunk in enumerate(relevant_docs):
            print(f"\n--- Chunk {i+1} ---\n{chunk[:250]}...")
    else:
        print("Tidak ada dokumen relevan yang ditemukan.")
    print("="*50 + "\n")