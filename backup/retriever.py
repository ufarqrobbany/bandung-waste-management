# import joblib
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import logging
# import os

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class DocumentRetriever:
#     """
#     Kelas untuk memuat data dokumen yang telah diproses dan mengambil bagian (chunk) 
#     yang paling relevan terhadap query menggunakan TF-IDF dan cosine similarity.
#     """

#     def __init__(self, data_path="perda_data.pkl"):
#         """
#         Inisialisasi DocumentRetriever.

#         Args:
#             data_path (str): Path ke file pickle yang berisi data dokumen, vectorizer, dan TF-IDF matrix.
#         """
#         self.data_path = data_path
#         self.chunks = []
#         self.vectorizer = None
#         self.tfidf_matrix = None
#         self._load_data()

#     def __str__(self):
#         """
#         Representasi string dari objek.

#         Returns:
#             str: Ringkasan jumlah chunks yang dimuat.
#         """
#         return f"<DocumentRetriever | chunks: {len(self.chunks)}>"

#     def _load_data(self):
#         """
#         Memuat data dari file pickle yang telah diproses sebelumnya.
#         File ini harus berisi kunci 'chunks', 'vectorizer', dan 'tfidf_matrix'.
#         Jika salah satu hilang, maka data dianggap tidak valid.
#         """
#         if not os.path.exists(self.data_path):
#             logging.error(f"File data tidak ditemukan: {self.data_path}. Jalankan perda_processor.py.")
#             return
        
#         try:
#             data = joblib.load(self.data_path)
#             self.chunks = data.get('chunks', [])
#             self.vectorizer = data.get('vectorizer')
#             self.tfidf_matrix = data.get('tfidf_matrix')
            
#             if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
#                 logging.error("Data yang dimuat tidak lengkap. Pastikan perda_data.pkl valid.")
#                 self.chunks = []  # Reset untuk mencegah penggunaan data tidak valid
#                 return

#             logging.info(f"Data retriever berhasil dimuat. Total chunks: {len(self.chunks)}")
#         except Exception as e:
#             logging.error(f"Gagal memuat data dari {self.data_path}: {e}")
#             self.chunks = []

#     def retrieve_chunks(self, query, top_k=3):
#         """
#         Mengambil potongan dokumen (chunks) yang paling relevan terhadap query yang diberikan
#         menggunakan cosine similarity terhadap representasi TF-IDF.

#         Args:
#             query (str): Pertanyaan atau masukan dari pengguna.
#             top_k (int): Jumlah top hasil paling relevan yang ingin dikembalikan.

#         Returns:
#             list[str]: Daftar chunks teks yang paling relevan terhadap query.
#         """
#         if not self.chunks or self.vectorizer is None:
#             logging.warning("Retriever tidak siap. Kembalikan array kosong.")
#             return []
            
#         if not query.strip():
#             logging.info("Query kosong. Tidak ada retrieval yang dilakukan.")
#             return []
        
#         query_vector = self.vectorizer.transform([query])
#         cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
#         top_k_indices = cosine_similarities.argsort()[-top_k:][::-1]
        
#         relevant_chunks = [self.chunks[i] for i in top_k_indices]
        
#         logging.info(f"Ditemukan {len(relevant_chunks)} chunks relevan untuk query.")
#         return relevant_chunks

# # Contoh penggunaan
# if __name__ == "__main__":
#     retriever = DocumentRetriever()
#     print(retriever)
    
#     test_query = "Bagaimana cara membuang sachet kopi?"
#     relevant_docs = retriever.retrieve_chunks(test_query, top_k=3)
    
#     print("\n--- Hasil Retrieval ---")
#     print(f"Query: {test_query}")
#     for i, chunk in enumerate(relevant_docs):
#         print(f"\nChunk {i+1}:\n{chunk[:250]}...")

import joblib
from sklearn.metrics.pairwise import cosine_similarity
import logging
import os
from sentence_transformers import CrossEncoder # <-- Tambahkan ini

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentRetriever:
    """
    Kelas untuk mengambil dokumen relevan.
    Tahap 1: Pengambilan cepat dengan TF-IDF.
    Tahap 2: Pemeringkatan ulang (reranking) dengan model semantik.
    """

    def __init__(self, data_path="perda_data.pkl"):
        """
        Inisialisasi DocumentRetriever.
        Memuat data TF-IDF dan model reranker.
        """
        self.data_path = data_path
        self.chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._load_data()

        # --- Tambahan Baru: Muat Model Reranker ---
        # Model ini akan dimuat sekali dan digunakan kembali.
        # 'cross-encoder/ms-marco-minilm-l-6-v2' adalah model yang ringan dan efektif.
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-minilm-l-6-v2')
            logging.info("Model CrossEncoder (reranker) berhasil dimuat.")
        except Exception as e:
            logging.error(f"Gagal memuat model CrossEncoder: {e}")
            self.reranker = None
        # ---------------------------------------------


    def __str__(self):
        return f"<DocumentRetriever | chunks: {len(self.chunks)} | Reranker Loaded: {self.reranker is not None}>"

    def _load_data(self):
        """
        Memuat data dari file pickle yang telah diproses sebelumnya.
        """
        if not os.path.exists(self.data_path):
            logging.error(f"File data tidak ditemukan: {self.data_path}. Jalankan perda_processor.py.")
            return
        
        try:
            data = joblib.load(self.data_path)
            self.chunks = data.get('chunks', [])
            self.vectorizer = data.get('vectorizer')
            self.tfidf_matrix = data.get('tfidf_matrix')
            
            if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
                logging.error("Data yang dimuat tidak lengkap. Pastikan perda_data.pkl valid.")
                self.chunks = []
                return

            logging.info(f"Data retriever (TF-IDF) berhasil dimuat. Total chunks: {len(self.chunks)}")
        except Exception as e:
            logging.error(f"Gagal memuat data dari {self.data_path}: {e}")
            self.chunks = []

    def retrieve_chunks(self, query, top_k=5):
        """
        Mengambil potongan dokumen (chunks) yang paling relevan.

        Args:
            query (str): Pertanyaan dari pengguna.
            top_k (int): Jumlah hasil akhir yang paling relevan yang ingin dikembalikan.

        Returns:
            list[str]: Daftar chunks teks yang paling relevan setelah di-rerank.
        """
        if not self.chunks or self.vectorizer is None:
            logging.warning("Retriever TF-IDF tidak siap. Mengembalikan list kosong.")
            return []
            
        if not query.strip():
            logging.info("Query kosong. Tidak ada retrieval yang dilakukan.")
            return []
        
        # --- Tahap 1: Retrieval Cepat dengan TF-IDF ---
        # Ambil lebih banyak kandidat dari TF-IDF (misal: 20) untuk diberikan ke reranker.
        initial_candidate_count = 20
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Ambil indeks dari kandidat teratas
        top_initial_indices = cosine_similarities.argsort()[-initial_candidate_count:][::-1]
        initial_chunks = [self.chunks[i] for i in top_initial_indices]
        
        logging.info(f"TF-IDF menemukan {len(initial_chunks)} kandidat awal.")
        
        # --- Tahap 2: Reranking Semantik ---
        if not self.reranker or not initial_chunks:
            # Jika reranker gagal dimuat atau tidak ada kandidat, kembalikan hasil TF-IDF
            logging.warning("Reranker tidak tersedia atau tidak ada hasil awal. Mengembalikan hasil TF-IDF.")
            return initial_chunks[:top_k]

        # Buat pasangan [query, chunk] untuk di-skor oleh reranker
        rerank_pairs = [[query, chunk] for chunk in initial_chunks]
        
        # Dapatkan skor relevansi dari model reranker
        scores = self.reranker.predict(rerank_pairs)
        
        # Gabungkan chunk dengan skornya, lalu urutkan
        scored_chunks = list(zip(scores, initial_chunks))
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Ambil kembali chunks yang sudah terurut
        reranked_chunks = [chunk for score, chunk in scored_chunks]

        logging.info(f"Reranker selesai memproses. Mengembalikan top {top_k} hasil.")
        
        # Kembalikan top-k hasil terbaik setelah di-rerank
        return reranked_chunks[:top_k]


# Contoh penggunaan
if __name__ == "__main__":
    retriever = DocumentRetriever()
    print(retriever)
    
    test_query = "Bagaimana cara membuang sachet kopi?"
    relevant_docs = retriever.retrieve_chunks(test_query, top_k=3)
    
    print("\n--- Hasil Retrieval (Setelah Reranking) ---")
    print(f"Query: {test_query}")
    for i, chunk in enumerate(relevant_docs):
        print(f"\nChunk {i+1}:\n{chunk[:250]}...")