import unittest
import sys
import os

# Menambahkan path src ke sys.path agar modul dapat diimpor
sys.path.append(os.path.abspath("src"))

from retriever import DocumentRetriever
from generator import LLMGenerator

class TestChatbotIntegration(unittest.TestCase):
    """
    Unit test untuk menguji integrasi end-to-end dari sistem chatbot
    yang melibatkan DocumentRetriever dan LLMGenerator.
    """

    def setUp(self):
        """
        Setup test dengan inisialisasi retriever dan generator.

        Mengasumsikan bahwa file 'perda_data.pkl' sudah tersedia
        setelah menjalankan perda_processor.py.
        """
        self.data_path = os.path.join(os.path.dirname(__file__), "..", "data", "perda_data.pkl")
        if not os.path.exists(self.data_path):
            self.fail(f"File data tidak ditemukan di {self.data_path}. Pastikan sudah dibuat.")

        self.retriever = DocumentRetriever(self.data_path)
        self.generator = LLMGenerator()

    def test_end_to_end_flow(self):
        """
        Menguji alur lengkap mulai dari pengambilan dokumen hingga
        pembuatan jawaban dari query yang diberikan.
        """
        query = "Apa sanksi bagi pembakar sampah?"

        # Tahap 1: Retrieval
        relevant_chunks = self.retriever.retrieve_chunks(query, top_k=1)
        self.assertTrue(relevant_chunks, "Retrieval seharusnya mengembalikan chunks yang relevan.")

        # Tahap 2: Generation
        response = self.generator.generate_answer(query, relevant_chunks)
        self.assertTrue(len(response) > 20, "Generator seharusnya mengembalikan respons yang informatif.")

if __name__ == "__main__":
    unittest.main()