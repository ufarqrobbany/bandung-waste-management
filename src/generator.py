import logging
from llama_cpp import Llama
import os

# Konfigurasi logging global dengan format standar
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMGenerator:
    """
    Kelas untuk menghasilkan jawaban dari pertanyaan pengguna dengan menggunakan
    model Large Language Model (LLM) lokal melalui pustaka llama-cpp-python.
    """

    # Jumlah maksimum token yang dapat ditangani oleh model dalam satu input prompt
    MAX_CONTEXT_TOKENS = 2048

    # Jumlah maksimum token yang diperbolehkan dari dokumen (chunks) untuk membentuk prompt
    MAX_CHUNK_TOKENS = 150

    def __init__(self):
        """
        Konstruktor untuk menginisialisasi model LLM dari file .gguf lokal.
        Memastikan file model tersedia sebelum memuatnya.
        """
        # Tentukan path model GGUF relatif terhadap file ini
        model_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "model",
            "qwen1_5-1_8b-chat-q4_k_m.gguf"
        )

        # Validasi keberadaan file model
        if not os.path.exists(model_path):
            logging.error(f"File model GGUF tidak ditemukan di: {model_path}. Mohon periksa lokasinya.")
            self.llm = None
            return

        try:
            # Inisialisasi model LLM dengan konfigurasi context window
            self.llm = Llama(model_path=model_path, n_ctx=self.MAX_CONTEXT_TOKENS, verbose=False)
            logging.info(f"Model Llama.cpp berhasil dimuat dengan context window {self.MAX_CONTEXT_TOKENS}. Generator siap.")
        except Exception as e:
            logging.error(f"Gagal memuat model Llama.cpp: {e}")
            self.llm = None

    def _create_prompt(self, query, retrieved_chunks):
        """
        Membuat prompt input untuk LLM dari query pengguna dan dokumen yang relevan.

        Parameters:
            query (str): Pertanyaan dari pengguna.
            retrieved_chunks (List[str]): List dokumen atau paragraf yang relevan.

        Returns:
            str: Prompt lengkap yang akan dikirim ke model LLM.
        """
        # Template dasar untuk sistem prompt
        prompt_template = (
            "Anda adalah asisten AI yang ahli dalam Peraturan Daerah Kota Bandung "
            "tentang Pengelolaan Sampah. Tugas Anda adalah memberikan jawaban yang singkat, "
            "sopan, dan berbasis fakta dari dokumen yang disediakan.\n\n"
            "Dokumen terkait:\n"
            "{chunks}\n\n"
            "Berdasarkan dokumen di atas, jawab pertanyaan pengguna berikut:\n"
            "Pertanyaan: {query}\n"
            "Jawaban:"
        )

        # Bangun konten dokumen yang dibatasi jumlah token-nya
        chunks_text_list = []
        current_token_count = 0
        for chunk in retrieved_chunks:
            chunk_tokens = len(chunk.split())
            if current_token_count + chunk_tokens <= self.MAX_CHUNK_TOKENS:
                chunks_text_list.append(chunk)
                current_token_count += chunk_tokens
            else:
                break  # Hentikan jika melebihi batas

        chunks_text = "\n---\n".join(chunks_text_list)
        prompt = prompt_template.format(chunks=chunks_text, query=query)

        return prompt

    def generate_answer(self, query, retrieved_chunks):
        """
        Menghasilkan jawaban atas pertanyaan pengguna berdasarkan dokumen yang relevan.

        Parameters:
            query (str): Pertanyaan dari pengguna.
            retrieved_chunks (List[str]): Dokumen relevan yang telah diambil dari retriever.

        Returns:
            str: Jawaban dari model LLM atau pesan kesalahan jika terjadi kegagalan.
        """
        # Validasi awal sebelum memproses prompt
        if self.llm is None:
            return "Maaf, generator tidak dapat memuat model LLM lokal."

        if not query.strip():
            return "Pertanyaan tidak boleh kosong. Mohon masukkan pertanyaan Anda."

        if not retrieved_chunks:
            return "Maaf, saya tidak dapat menemukan informasi yang relevan dalam dokumen."

        # Bangun prompt dari query dan chunks
        prompt_content = self._create_prompt(query, retrieved_chunks)

        # Validasi panjang prompt sebelum dikirim
        if len(prompt_content.split()) > self.MAX_CONTEXT_TOKENS:
            logging.error("Prompt masih terlalu panjang setelah pemotongan. Silakan kurangi ukuran chunks lebih lanjut.")
            return "Maaf, prompt yang dihasilkan terlalu panjang untuk model. Silakan coba pertanyaan yang lebih ringkas."

        logging.info("Prompt berhasil dibuat. Menghasilkan jawaban dengan model LLM...")

        try:
            # Kirim prompt ke model untuk menghasilkan jawaban
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Anda adalah asisten AI yang ahli dalam Peraturan Daerah "
                            "Kota Bandung tentang Pengelolaan Sampah."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt_content
                    }
                ],
                stop=["Jawaban:", "###"],  # Pemicu untuk menghentikan output
                max_tokens=256,             # Batas panjang jawaban
                temperature=0.7             # Kontrol variasi jawaban
            )

            return response['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"Gagal menghasilkan respons dari model: {e}")
            return "Maaf, terjadi kesalahan saat menghubungi model LLM lokal."