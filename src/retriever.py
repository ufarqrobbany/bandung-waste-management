import os
import joblib
import logging
from typing import List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DocumentRetriever:
    """
    Kelas untuk memuat data dokumen yang telah diproses dan mengambil bagian (chunk) 
    yang paling relevan terhadap query menggunakan TF-IDF dan cosine similarity.
    Ini adalah implementasi baseline tanpa semantic reranking.
    """

    def __init__(self, data_path: str = "data/perda_data.pkl"):
        """
        Inisialisasi DocumentRetriever.

        Args:
            data_path (str): Path ke file pickle yang berisi data dokumen, vectorizer, dan TF-IDF matrix.
        """
        self.data_path = data_path
        self.chunks: List[str] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None
        self._load_data()

    def __str__(self) -> str:
        """Representasi string dari objek."""
        return f"<DocumentRetriever | chunks: {len(self.chunks)}>"

    def _load_data(self):
        """
        Memuat data dari file pickle yang telah diproses sebelumnya.
        File ini harus berisi kunci 'chunks', 'vectorizer', dan 'tfidf_matrix'.
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
                self.chunks = []
                self.vectorizer = None
                self.tfidf_matrix = None
                return

            logging.info(f"Data retriever berhasil dimuat. Total chunks: {len(self.chunks)}")
        except Exception as e:
            logging.error(f"Gagal memuat data dari {self.data_path}: {e}")
            self.chunks = []
            self.vectorizer = None
            self.tfidf_matrix = None

    def retrieve_chunks(self, query: str, top_k: int = 10) -> List[str]:
        """
        Mengambil potongan dokumen (chunks) yang paling relevan terhadap query yang diberikan
        menggunakan cosine similarity terhadap representasi TF-IDF.

        Args:
            query (str): Pertanyaan atau masukan dari pengguna.
            top_k (int): Jumlah hasil paling relevan yang ingin dikembalikan.

        Returns:
            List[str]: Daftar chunks teks yang paling relevan terhadap query.
        """
        if not self.chunks or self.vectorizer is None or self.tfidf_matrix is None:
            logging.warning("Retriever tidak siap. Kembalikan array kosong.")
            return []
            
        if not query.strip():
            logging.info("Query kosong. Tidak ada retrieval yang dilakukan.")
            return []
        
        try:
            query_vector = self.vectorizer.transform([query])
            cosine_similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            top_k_indices = cosine_similarities.argsort()[-top_k:][::-1]
            
            relevant_chunks = [self.chunks[i] for i in top_k_indices]
            
            logging.info(f"Ditemukan {len(relevant_chunks)} chunks relevan untuk query.")
            return relevant_chunks
        except Exception as e:
            logging.error(f"Gagal melakukan retrieval: {e}")
            return []

# Contoh penggunaan
if __name__ == "__main__":
    # Catatan: File data/perda_data.pkl harus sudah dibuat menggunakan skrip pemrosesan data.
    retriever = DocumentRetriever()
    print(retriever)
    
    test_query = "Bagaimana cara membuang sachet kopi?"
    relevant_docs = retriever.retrieve_chunks(test_query, top_k=10)
    
    print("\n--- Hasil Retrieval (TF-IDF Baseline) ---")
    print(f"Query: {test_query}")
    for i, chunk in enumerate(relevant_docs):
        print(f"\nChunk {i+1}:\n{chunk[:250]}...")