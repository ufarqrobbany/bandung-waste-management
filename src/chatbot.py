# import argparse
# import logging
# from retriever import DocumentRetriever
# from generator import LLMGenerator

# # Konfigurasi logging untuk menampilkan log pada terminal
# # Format log: timestamp - level - pesan
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# def main():
#     """
#     Fungsi utama untuk menjalankan chatbot berbasis Command-Line Interface (CLI).
#     Proses kerja meliputi:
#         1. Parsing argumen dari command line.
#         2. Memuat model retriever dan generator.
#         3. Mengambil dokumen relevan berdasarkan pertanyaan pengguna.
#         4. Menghasilkan jawaban menggunakan model generatif.
#         5. Menampilkan jawaban dan referensi dokumen yang digunakan.
#     """
    
#     # Inisialisasi parser argumen
#     parser = argparse.ArgumentParser(description='Chatbot RAG edukasi sampah berbasis PERDA.')
    
#     # Argumen utama: pertanyaan dari pengguna
#     parser.add_argument('query', type=str, help='Pertanyaan untuk chatbot.')
    
#     # Argumen opsional: path ke file data (default: perda_data.pkl)
#     parser.add_argument('--data-path', type=str, default="perda_data.pkl", help='Path file data yang diproses.')
    
#     # Parsing argumen dari command-line
#     args = parser.parse_args()

#     # Inisialisasi DocumentRetriever
#     logging.info("Menginisialisasi DocumentRetriever...")
#     try:
#         retriever = DocumentRetriever(data_path=args.data_path)

#         # Validasi apakah data berhasil dimuat
#         if not retriever.chunks:
#             logging.error("Retriever tidak dapat memuat data. Mohon jalankan 'perda_processor.py' terlebih dahulu.")
#             return
#     except Exception as e:
#         logging.error(f"Gagal menginisialisasi retriever: {e}")
#         return

#     # Inisialisasi LLMGenerator
#     logging.info("Menginisialisasi LLMGenerator...")
#     generator = LLMGenerator()

#     # Proses retrieval: mengambil top-k dokumen relevan dari query
#     logging.info(f"Mencari informasi relevan untuk query: '{args.query}'")
#     retrieved_chunks = retriever.retrieve_chunks(args.query, top_k=3)

#     # Proses generation: menghasilkan jawaban dari dokumen yang diambil
#     logging.info("Menghasilkan jawaban berdasarkan dokumen yang ditemukan...")
#     final_answer = generator.generate_answer(args.query, retrieved_chunks)

#     # Output jawaban akhir ke terminal
#     print("\n" + "="*50)
#     print("Jawaban Chatbot:")
#     print(final_answer)
    
#     # Output referensi dokumen (jika ada)
#     if retrieved_chunks:
#         print("\n" + "="*50)
#         print("Referensi Dokumen:")
#         for i, chunk in enumerate(retrieved_chunks):
#             # Menampilkan sebagian isi chunk untuk referensi pengguna
#             print(f"\n--- Chunk {i+1} ---")
#             print(f"{chunk[:250]}...")
#     print("="*50 + "\n")

# # Entry point program
# if __name__ == "__main__":
#     main()

import argparse
import logging
from retriever import DocumentRetriever
from generator import LLMGeneratorSync as LLMGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Chatbot RAG edukasi sampah berbasis PERDA.')
    parser.add_argument('query', type=str, help='Pertanyaan untuk chatbot.')
    parser.add_argument('--data-path', type=str, default="data/perda_data.pkl", help='Path file data.')
    args = parser.parse_args()

    logging.info("Menginisialisasi DocumentRetriever...")
    retriever = DocumentRetriever(data_path=args.data_path)
    if not retriever.chunks:
        logging.error("Gagal memuat data retriever. Jalankan 'perda_processor.py' dulu.")
        return

    logging.info("Menginisialisasi LLMGenerator...")
    generator = LLMGenerator()

    # --- PERUBAHAN DI SINI ---
    logging.info(f"Mencari informasi relevan untuk query: '{args.query}'")
    # retrieved_results sekarang berisi (chunk, score)
    retrieved_results = retriever.retrieve_chunks(args.query, top_k=3)

    # Ekstrak hanya teks chunk untuk dikirim ke generator
    retrieved_chunks = [result[0] for result in retrieved_results] if retrieved_results else []

    logging.info("Menghasilkan jawaban...")
    final_answer = generator.generate_answer(args.query, retrieved_chunks)

    print("\n" + "="*50)
    print("Jawaban Chatbot:")
    print(final_answer)
    
    if retrieved_results:
        print("\n" + "="*50)
        print("Referensi Dokumen (Hasil Reranking):")
        # Tampilkan chunk beserta skornya
        for i, (chunk, score) in enumerate(retrieved_results):
            print(f"\n--- Chunk {i+1} | Skor Relevansi: {score:.4f} ---")
            print(f"{chunk[:250]}...")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()