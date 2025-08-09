import logging
from groq import Groq, AsyncGroq
import groq
from config import AppConfig

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMGeneratorSync:
    """
    Kelas untuk menghasilkan jawaban menggunakan model remote (Groq API) secara sinkron.
    """
    def __init__(self):
        """
        Konstruktor untuk menginisialisasi client Groq dan konfigurasi.
        """
        self.config = AppConfig()
        if not self.config.GROQ_API_KEY:
            logging.error("API key tidak ditemukan. Pastikan GROQ_API_KEY telah diatur.")
            self.client = None
            return

        try:
            self.client = Groq(api_key=self.config.GROQ_API_KEY)
            logging.info("Generator Groq sinkron berhasil diinisialisasi.")
        except Exception as e:
            logging.error(f"Gagal menginisialisasi Groq client: {e}")
            self.client = None

    def _create_prompt_messages(self, query, retrieved_chunks):
        """Mempersiapkan pesan dalam format yang dibutuhkan oleh API."""
        context = "\n---\n".join(retrieved_chunks)
        user_content = (
            f"Konteks Dokumen:\n"
            f"-----------------\n"
            f"{context}\n"
            f"-----------------\n\n"
            f"Berdasarkan konteks di atas, jawab pertanyaan ini:\n"
            f"Pertanyaan: {query}"
        )

        return [
            {"role": "system", "content": self.config.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

    def generate_answer(self, query, retrieved_chunks):
        """
        Menghasilkan jawaban dengan memanggil API Groq secara sinkron.
        """
        if self.client is None:
            return "Maaf, generator tidak terhubung ke layanan LLM remote."

        if not query.strip():
            return "Pertanyaan tidak boleh kosong."

        if not retrieved_chunks:
            return "Maaf, tidak ada informasi relevan dalam dokumen untuk menjawab pertanyaan Anda."

        logging.info(f"Membuat prompt untuk pertanyaan: '{query[:50]}...'")
        messages = self._create_prompt_messages(query, retrieved_chunks)
        logging.info("Mengirim permintaan ke Groq API...")

        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.config.LLM_MODEL,
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.LLM_MAX_TOKENS
            )
            response_content = chat_completion.choices[0].message.content
            logging.info("Respons dari Groq API berhasil diterima.")
            return response_content
        except groq.APIError as e:
            logging.error(f"Groq API Error: {e}")
            return "Maaf, terjadi kesalahan pada API. Silakan coba lagi."
        except groq.RateLimitError as e:
            logging.warning(f"Groq Rate Limit Error: {e}")
            return "Layanan sedang sibuk. Mohon tunggu sebentar dan coba lagi."
        except Exception as e:
            logging.error(f"Gagal mendapatkan respons dari Groq API: {e}")
            return "Maaf, terjadi kesalahan yang tidak terduga."

class LLMGeneratorAsync:
    """
    Kelas untuk menghasilkan jawaban menggunakan model remote (Groq API) secara asinkron.
    """
    def __init__(self):
        self.config = AppConfig()
        if not self.config.GROQ_API_KEY:
            logging.error("API key tidak ditemukan. Pastikan GROQ_API_KEY telah diatur.")
            self.client = None
            return
        
        try:
            self.client = AsyncGroq(api_key=self.config.GROQ_API_KEY)
            logging.info("Generator Groq asinkron berhasil diinisialisasi.")
        except Exception as e:
            logging.error(f"Gagal menginisialisasi Groq client: {e}")
            self.client = None

    def _create_prompt_messages(self, query, retrieved_chunks):
        """Mempersiapkan pesan dalam format yang dibutuhkan oleh API."""
        # Logika sama seperti versi sinkron
        context = "\n---\n".join(retrieved_chunks)
        user_content = (
            f"Konteks Dokumen:\n"
            f"-----------------\n"
            f"{context}\n"
            f"-----------------\n\n"
            f"Berdasarkan konteks di atas, jawab pertanyaan ini:\n"
            f"Pertanyaan: {query}"
        )
        return [
            {"role": "system", "content": self.config.SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

    async def generate_answer(self, query, retrieved_chunks):
        """
        Menghasilkan jawaban dengan memanggil API Groq secara asinkron.
        """
        if self.client is None:
            return "Maaf, generator tidak terhubung ke layanan LLM remote."

        if not query.strip():
            return "Pertanyaan tidak boleh kosong."

        if not retrieved_chunks:
            return "Maaf, tidak ada informasi relevan dalam dokumen untuk menjawab pertanyaan Anda."

        logging.info(f"Membuat prompt untuk pertanyaan: '{query[:50]}...'")
        messages = self._create_prompt_messages(query, retrieved_chunks)
        logging.info("Mengirim permintaan asinkron ke Groq API...")

        try:
            chat_completion = await self.client.chat.completions.create(
                messages=messages,
                model=self.config.LLM_MODEL,
                temperature=self.config.LLM_TEMPERATURE,
                max_tokens=self.config.LLM_MAX_TOKENS
            )
            response_content = chat_completion.choices[0].message.content
            logging.info("Respons dari Groq API berhasil diterima.")
            return response_content
        except groq.APIError as e:
            logging.error(f"Groq API Error: {e}")
            return "Maaf, terjadi kesalahan pada API. Silakan coba lagi."
        except groq.RateLimitError as e:
            logging.warning(f"Groq Rate Limit Error: {e}")
            return "Layanan sedang sibuk. Mohon tunggu sebentar dan coba lagi."
        except Exception as e:
            logging.error(f"Gagal mendapatkan respons dari Groq API: {e}")
            return "Maaf, terjadi kesalahan yang tidak terduga."