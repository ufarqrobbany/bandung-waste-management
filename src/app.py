import streamlit as st
import joblib
import logging
import asyncio 
from retriever import DocumentRetriever
from generator import LLMGeneratorAsync as LLMGenerator
from translate import Translator

# -------------------------
# Logging Configuration
# -------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------
# Initialize Translators
# -------------------------
try:
    translator_id_to_en = Translator(to_lang="en", from_lang="id")
    translator_en_to_id = Translator(to_lang="id", from_lang="en")
except Exception as e:
    st.error(f"Gagal menginisialisasi translator: {e}. Pastikan Anda memiliki koneksi internet.")
    translator_id_to_en = None
    translator_en_to_id = None

# -------------------------
# Component Loader Function
# -------------------------
@st.cache_resource
def load_components():
    """
    Memuat komponen utama:
    - Retriever: untuk mengambil konteks dari dokumen
    - Generator: untuk menghasilkan jawaban
    - Classifier: untuk klasifikasi jenis pertanyaan/sampah

    Returns:
        Tuple (retriever, generator, classifier)
    """
    logging.info("Memuat komponen: Retriever, Generator, dan Classifier...")
    try:
        retriever = DocumentRetriever(data_path="data/perda_data.pkl")
        generator = LLMGenerator()
        classifier = joblib.load("data/classifier_model.pkl")
        return retriever, generator, classifier
    except FileNotFoundError as e:
        st.error(f"Gagal memuat file: {e}. Pastikan file data/perda_data.pkl dan data/classifier_model.pkl ada.")
        return None, None, None
    except Exception as e:
        st.error(f"Gagal memuat komponen: {e}")
        return None, None, None

# Load komponen hanya satu kali
retriever, generator, classifier = load_components()

# -------------------------
# Main Chatbot Function
# -------------------------
# Ubah fungsi menjadi async
async def run_chatbot_async(query):
    """
    Menjalankan pipeline chatbot secara asinkron.
    """
    if not retriever or not generator or not classifier:
        st.error("Komponen chatbot tidak berhasil dimuat. Mohon periksa log.")
        return

    if not query:
        st.error("Mohon masukkan pertanyaan.")
        return

    st.info("Sedang memproses pertanyaan...")

    # --- Step 1: Retrieval chunk dokumen ---
    retrieved_chunks = retriever.retrieve_chunks(query)

    # --- Step 2: Generasi jawaban ---
    answer = await generator.generate_answer(query, retrieved_chunks)

    # --- Output ke pengguna ---
    st.markdown("---")
    st.markdown(f"**Jawaban:**\n{answer}")

    # Tampilkan referensi chunk yang digunakan
    if retrieved_chunks:
        st.markdown("\n**Referensi Dokumen:**")
        for i, chunk in enumerate(retrieved_chunks):
            st.text(f"Chunk {i+1}: {chunk[:200]}...")

# -------------------------
# Streamlit UI Layout
# -------------------------
st.title("Chatbot Edukasi Pengelolaan Sampah")
st.write("Silakan ajukan pertanyaan tentang pengelolaan sampah sesuai PERDA Kota Bandung.")

# Input pertanyaan dari user
query = st.text_input("Tulis pertanyaan Anda di sini:", key="user_query")

# Tombol untuk mengirim pertanyaan ke chatbot
if st.button("Kirim"):
    # Jalankan fungsi asinkron di dalam event loop
    asyncio.run(run_chatbot_async(query))