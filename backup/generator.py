# import logging
# from llama_cpp import Llama
# import os

# # Konfigurasi logging global dengan format standar
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# class LLMGenerator:
#     """
#     Kelas untuk menghasilkan jawaban dari pertanyaan pengguna dengan menggunakan
#     model Large Language Model (LLM) lokal melalui pustaka llama-cpp-python.
#     """

#     # Jumlah maksimum token yang dapat ditangani oleh model dalam satu input prompt
#     MAX_CONTEXT_TOKENS = 2048

#     # Jumlah maksimum token yang diperbolehkan dari dokumen (chunks) untuk membentuk prompt
#     MAX_CHUNK_TOKENS = 150

#     def __init__(self):
#         """
#         Konstruktor untuk menginisialisasi model LLM dari file .gguf lokal.
#         Memastikan file model tersedia sebelum memuatnya.
#         """
#         # Tentukan path model GGUF relatif terhadap file ini
#         model_path = os.path.join(
#             os.path.dirname(__file__),
#             "..",
#             "model",
#             "qwen1_5-1_8b-chat-q4_k_m.gguf"
#         )

#         # Validasi keberadaan file model
#         if not os.path.exists(model_path):
#             logging.error(f"File model GGUF tidak ditemukan di: {model_path}. Mohon periksa lokasinya.")
#             self.llm = None
#             return

#         try:
#             # Inisialisasi model LLM dengan konfigurasi context window
#             self.llm = Llama(model_path=model_path, n_ctx=self.MAX_CONTEXT_TOKENS, verbose=False)
#             logging.info(f"Model Llama.cpp berhasil dimuat dengan context window {self.MAX_CONTEXT_TOKENS}. Generator siap.")
#         except Exception as e:
#             logging.error(f"Gagal memuat model Llama.cpp: {e}")
#             self.llm = None

#     def _create_prompt(self, query, retrieved_chunks):
#         """
#         Membuat prompt input untuk LLM dari query pengguna dan dokumen yang relevan.

#         Parameters:
#             query (str): Pertanyaan dari pengguna.
#             retrieved_chunks (List[str]): List dokumen atau paragraf yang relevan.

#         Returns:
#             str: Prompt lengkap yang akan dikirim ke model LLM.
#         """
#         # Template dasar untuk sistem prompt
#         prompt_template = (
#             "Anda adalah asisten AI yang ahli dalam Peraturan Daerah Kota Bandung "
#             "tentang Pengelolaan Sampah. Tugas Anda adalah memberikan jawaban yang singkat, "
#             "sopan, dan berbasis fakta dari dokumen yang disediakan.\n\n"
#             "Dokumen terkait:\n"
#             "{chunks}\n\n"
#             "Berdasarkan dokumen di atas, jawab pertanyaan pengguna berikut:\n"
#             "Pertanyaan: {query}\n"
#             "Jawaban:"
#         )

#         # Bangun konten dokumen yang dibatasi jumlah token-nya
#         chunks_text_list = []
#         current_token_count = 0
#         for chunk in retrieved_chunks:
#             chunk_tokens = len(chunk.split())
#             if current_token_count + chunk_tokens <= self.MAX_CHUNK_TOKENS:
#                 chunks_text_list.append(chunk)
#                 current_token_count += chunk_tokens
#             else:
#                 break  # Hentikan jika melebihi batas

#         chunks_text = "\n---\n".join(chunks_text_list)
#         prompt = prompt_template.format(chunks=chunks_text, query=query)

#         return prompt

#     def generate_answer(self, query, retrieved_chunks):
#         """
#         Menghasilkan jawaban atas pertanyaan pengguna berdasarkan dokumen yang relevan.

#         Parameters:
#             query (str): Pertanyaan dari pengguna.
#             retrieved_chunks (List[str]): Dokumen relevan yang telah diambil dari retriever.

#         Returns:
#             str: Jawaban dari model LLM atau pesan kesalahan jika terjadi kegagalan.
#         """
#         # Validasi awal sebelum memproses prompt
#         if self.llm is None:
#             return "Maaf, generator tidak dapat memuat model LLM lokal."

#         if not query.strip():
#             return "Pertanyaan tidak boleh kosong. Mohon masukkan pertanyaan Anda."

#         if not retrieved_chunks:
#             return "Maaf, saya tidak dapat menemukan informasi yang relevan dalam dokumen."

#         # Bangun prompt dari query dan chunks
#         prompt_content = self._create_prompt(query, retrieved_chunks)

#         # Validasi panjang prompt sebelum dikirim
#         if len(prompt_content.split()) > self.MAX_CONTEXT_TOKENS:
#             logging.error("Prompt masih terlalu panjang setelah pemotongan. Silakan kurangi ukuran chunks lebih lanjut.")
#             return "Maaf, prompt yang dihasilkan terlalu panjang untuk model. Silakan coba pertanyaan yang lebih ringkas."

#         logging.info("Prompt berhasil dibuat. Menghasilkan jawaban dengan model LLM...")

#         try:
#             # Kirim prompt ke model untuk menghasilkan jawaban
#             response = self.llm.create_chat_completion(
#                 messages=[
#                     {
#                         "role": "system",
#                         "content": (
#                             "Anda adalah asisten AI yang ahli dalam Peraturan Daerah "
#                             "Kota Bandung tentang Pengelolaan Sampah."
#                         )
#                     },
#                     {
#                         "role": "user",
#                         "content": prompt_content
#                     }
#                 ],
#                 stop=["Jawaban:", "###"],  # Pemicu untuk menghentikan output
#                 max_tokens=256,             # Batas panjang jawaban
#                 temperature=0.7             # Kontrol variasi jawaban
#             )

#             return response['choices'][0]['message']['content']
#         except Exception as e:
#             logging.error(f"Gagal menghasilkan respons dari model: {e}")
#             return "Maaf, terjadi kesalahan saat menghubungi model LLM lokal."

import os
import logging
from groq import Groq

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMGenerator:
    """
    Kelas untuk menghasilkan jawaban menggunakan model remote (Groq API).
    """

    def __init__(self):
        """
        Konstruktor untuk menginisialisasi client Groq.
        Mengambil API key dari environment variable.
        """
        try:
            # Mengambil API key dari environment variable 'GROQ_API_KEY'
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                logging.error("Environment variable GROQ_API_KEY tidak ditemukan.")
                raise ValueError("Set GROQ_API_KEY environment variable")

            # Inisialisasi client Groq
            self.client = Groq(api_key=api_key)
            logging.info("Generator dengan Groq API berhasil diinisialisasi.")

        except Exception as e:
            logging.error(f"Gagal menginisialisasi Groq client: {e}")
            self.client = None

    def _create_prompt_messages(self, query, retrieved_chunks):
        """
        Mempersiapkan pesan dalam format yang dibutuhkan oleh API.
        Ini memisahkan instruksi sistem dari query pengguna.
        
        Returns:
            list: Daftar pesan untuk API.
        """
        # Gabungkan semua potongan teks yang relevan menjadi satu konteks
        context = "\n---\n".join(retrieved_chunks)

        # Instruksi untuk sistem (AI)
        system_prompt = (
            "Anda adalah asisten AI yang ahli dalam Peraturan Daerah Kota Bandung "
            "tentang Pengelolaan Sampah. Tugas Anda adalah memberikan jawaban yang singkat, "
            "sopan, dan berbasis fakta dari konteks dokumen yang disediakan. "
            "Jawablah hanya berdasarkan informasi dari konteks di bawah."
        )

        # Konten yang diberikan oleh pengguna (query + konteks)
        user_content = (
            f"Konteks Dokumen:\n"
            f"-----------------\n"
            f"{context}\n"
            f"-----------------\n\n"
            f"Berdasarkan konteks di atas, jawab pertanyaan ini:\n"
            f"Pertanyaan: {query}"
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        return messages

    def generate_answer(self, query, retrieved_chunks):
        """
        Menghasilkan jawaban dengan memanggil API Groq.
        """
        if self.client is None:
            return "Maaf, generator tidak dapat terhubung ke layanan LLM remote."

        if not query.strip():
            return "Pertanyaan tidak boleh kosong."

        if not retrieved_chunks:
            return "Maaf, saya tidak dapat menemukan informasi yang relevan dalam dokumen untuk menjawab pertanyaan Anda."

        # Buat pesan untuk dikirim ke API
        messages = self._create_prompt_messages(query, retrieved_chunks)

        logging.info("Mengirim permintaan ke Groq API...")

        try:
            # Panggil API untuk menghasilkan jawaban
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="llama3-8b-8192",  # Model yang digunakan di Groq, bisa diganti
                temperature=0.7,
                max_tokens=512
            )
            
            return chat_completion.choices[0].message.content
        
        except Exception as e:
            logging.error(f"Gagal mendapatkan respons dari Groq API: {e}")
            return "Maaf, terjadi kesalahan saat menghubungi layanan LLM remote."