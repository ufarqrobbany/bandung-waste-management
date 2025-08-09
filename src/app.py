import streamlit as st
import logging
import asyncio 
from retriever import DocumentRetriever
from generator import LLMGeneratorAsync as LLMGenerator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- KAMUS KONFIGURASI VERSI ---
# Di sini Anda bisa mendefinisikan semua versi yang ingin diuji.
RETRIEVER_MODES = {
    "Baseline (TF-IDF Saja)": {
        "use_reranker": False,
        "top_k": 5,
        "initial_k": 5 # Hanya butuh 5 karena tidak ada reranking
    },
    "Reranker (Seimbang)": {
        "use_reranker": True,
        "top_k": 5,
        "initial_k": 50
    },
    "Reranker (Akurasi Tinggi)": {
        "use_reranker": True,
        "top_k": 5,
        "initial_k": 200
    },
    "Reranker (Cepat)": {
        "use_reranker": True,
        "top_k": 3,
        "initial_k": 20
    }
}

@st.cache_resource
def load_components():
    """Memuat komponen retriever dan generator yang akan digunakan bersama."""
    logging.info("Memuat komponen: Retriever dan Generator...")
    try:
        retriever = DocumentRetriever(data_path="data/perda_data.pkl")
        generator = LLMGenerator()
        return retriever, generator
    except Exception as e:
        st.error(f"Gagal memuat komponen: {e}")
        return None, None

retriever, generator = load_components()

async def run_chatbot_async(query, mode_config):
    """Menjalankan pipeline chatbot menggunakan konfigurasi yang dipilih."""
    if not retriever or not generator:
        st.error("Komponen chatbot tidak berhasil dimuat.")
        return

    if not query:
        st.warning("Mohon masukkan pertanyaan.")
        return

    with st.spinner("Mencari dokumen relevan dan menghasilkan jawaban..."):
        # 1. Retrieval dengan parameter dinamis dari mode_config
        retrieved_results = retriever.retrieve_chunks(
            query, 
            top_k=mode_config["top_k"],
            initial_k=mode_config["initial_k"],
            use_reranker=mode_config["use_reranker"]
        )

        retrieved_chunks = [result[0] for result in retrieved_results] if retrieved_results else []

        # 2. Generasi jawaban
        answer = await generator.generate_answer(query, retrieved_chunks)

        st.markdown("---")
        st.markdown(f"**Jawaban:**\n{answer}")

        # 3. Tampilkan referensi dan skornya
        if retrieved_results:
            score_type = "Relevansi (Reranker)" if mode_config["use_reranker"] else "Relevansi (TF-IDF)"
            st.markdown(f"\n**Referensi Dokumen (Metode: {score_type}):**")
            for i, (chunk, score) in enumerate(retrieved_results):
                with st.expander(f"Referensi {i+1} | Skor: {score:.4f}"):
                    st.markdown(f"_{chunk}_")


# --- UI Layout Streamlit ---
st.title("🤖 Chatbot Edukasi Pengelolaan Sampah")
st.write("Ajukan pertanyaan tentang pengelolaan sampah sesuai PERDA Kota Bandung.")

# --- Panel Samping (Sidebar) untuk Pilihan Versi ---
st.sidebar.title("⚙️ Pengaturan Versi")
mode_selection = st.sidebar.selectbox(
    "Pilih versi retriever untuk diuji:",
    options=list(RETRIEVER_MODES.keys())
)

# Ambil konfigurasi yang dipilih
selected_config = RETRIEVER_MODES[mode_selection]

# Tampilkan detail konfigurasi yang sedang aktif di sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Konfigurasi Aktif:**")
st.sidebar.markdown(f"🔹 **Reranker Aktif:** `{selected_config['use_reranker']}`")
st.sidebar.markdown(f"🔹 **Hasil Akhir (top_k):** `{selected_config['top_k']}`")
if selected_config['use_reranker']:
    st.sidebar.markdown(f"🔹 **Kandidat Awal (initial_k):** `{selected_config['initial_k']}`")
st.sidebar.markdown("---")


# Input dari pengguna
query = st.text_input("Tulis pertanyaan Anda di sini:", key="user_query")

# Tombol kirim
if st.button("Kirim", type="primary"):
    if query:
        # Jalankan chatbot dengan konfigurasi yang dipilih dari sidebar
        asyncio.run(run_chatbot_async(query, selected_config))