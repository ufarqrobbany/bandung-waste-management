# --- Library Eksternal ---
import fitz # PyMuPDF untuk ekstraksi teks dari PDF
import re # Regex untuk pembersihan teks
import joblib # Untuk serialisasi model atau data
import nltk # Natural Language Toolkit untuk tokenisasi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer # TF-IDF untuk indexing teks
import os
import argparse # Untuk parsing argumen CLI
import logging # Logging untuk pelacakan proses
import numpy as np
from typing import List, Tuple, Optional

# --- Konfigurasi Logging Default ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pastikan stopwords Bahasa Indonesia tersedia
try:
    stopwords_id = set(stopwords.words('indonesian'))
except LookupError:
    logging.warning("NLTK 'stopwords' resource not found. Downloading...")
    nltk.download('stopwords')
    stopwords_id = set(stopwords.words('indonesian'))

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Mengekstrak teks dari file PDF menggunakan PyMuPDF (fitz).
    
    Args:
        pdf_path (str): Path ke file PDF.
    
    Returns:
        str | None: Seluruh teks dari PDF, atau None jika gagal.
    """
    try:
        if not os.path.exists(pdf_path):
            logging.error(f"File tidak ditemukan di: {pdf_path}")
            return None
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            page_text = page.get_text("text")
            if page_text:
                text += page_text
        doc.close()
        if not text.strip():
            logging.warning("Dokumen PDF tampaknya kosong atau tidak berisi teks yang dapat diekstrak.")
            return None
        return text
    except Exception as e:
        logging.error(f"Error saat memproses PDF: {e}")
        return None

def preprocess_text(text: str) -> str:
    """
    Membersihkan teks mentah dari hasil ekstraksi PDF.
    - Menghapus header/footer, footnote
    - Menghapus referensi hukum yang berulang
    - Mengubah ke lowercase, menghapus spasi berlebih, dll.
    - Menghapus stopwords
    
    Args:
        text (str): Teks mentah.
    
    Returns:
        str: Teks yang telah dibersihkan.
    """
    # Menghapus header/footer dan URL
    text = re.sub(r'WALI KOTA BANDUNG.*?\n', '', text, flags=re.DOTALL)
    text = re.sub(r'https://jdih\.bandung\.go\.id/.*?\n', '', text)
    text = re.sub(r'LEMBARAN DAERAH KOTA BANDUNG.*', '', text)
    text = re.sub(r'Penjelasan\nAtas.*', '', text, flags=re.DOTALL) # Menghapus penjelasan di akhir dokumen
    
    # Menghilangkan baris kosong dan spasi berlebih
    text = re.sub(r'\n\s*\n', '\n', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Mengubah ke lowercase dan menghapus karakter non-alfanumerik
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s.,]', '', text)
    
    # Menghapus stopwords (opsional, dapat diaktifkan/dinonaktifkan)
    # tokens = word_tokenize(text)
    # filtered_tokens = [word for word in tokens if word not in stopwords_id]
    # text = ' '.join(filtered_tokens)
    
    return text

def chunk_text_by_structure(text: str) -> List[str]:
    """
    Membagi teks menjadi chunks berdasarkan struktur dokumen (Bab, Pasal).
    
    Args:
        text (str): Teks yang sudah dibersihkan.
    
    Returns:
        List[str]: List berisi potongan teks (chunk) per struktur.
    """
    # Pola regex untuk menemukan judul Bab dan Pasal
    structure_pattern = r'(BAB\s+[IVXLCDM]+\s+|Pasal\s+\d+)'
    
    # Memisahkan teks berdasarkan pola tersebut
    parts = re.split(structure_pattern, text)
    
    chunks = []
    current_chunk_title = ""
    current_chunk_content = ""
    
    for part in parts:
        if re.match(structure_pattern, part):
            # Jika ini judul baru, simpan chunk sebelumnya jika ada
            if current_chunk_content:
                chunks.append(f"{current_chunk_title.strip()} {current_chunk_content.strip()}")
            current_chunk_title = part
            current_chunk_content = ""
        else:
            current_chunk_content += part
    
    # Simpan chunk terakhir
    if current_chunk_content:
        chunks.append(f"{current_chunk_title.strip()} {current_chunk_content.strip()}")

    # Jika struktur tidak terdeteksi, fallback ke chunking berbasis token
    if len(chunks) < 2:
        logging.warning("Struktur dokumen tidak terdeteksi. Menggunakan chunking berbasis token.")
        return chunk_text_by_token(text, chunk_size=300, overlap=50)
        
    return chunks

def chunk_text_by_token(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Membagi teks panjang menjadi beberapa potongan (chunk) dengan overlap token.
    Ini adalah metode fallback jika chunking struktural gagal.
    
    Args:
        text (str): Teks yang ingin dipecah.
    
    Returns:
        List[str]: List berisi potongan teks (chunk).
    """
    try:
        tokens = word_tokenize(text)
    except LookupError:
        logging.warning("NLTK 'punkt' resource not found. Downloading...")
        nltk.download('punkt')
        tokens = word_tokenize(text)
    
    chunks = []
    
    if chunk_size <= overlap:
        logging.error("chunk_size harus lebih besar dari overlap.")
        return []

    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = ' '.join(tokens[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def create_tfidf_index(chunks: List[str]) -> Tuple[TfidfVectorizer, np.ndarray]:
    """
    Membuat indeks TF-IDF dari daftar chunk teks.
    
    Args:
        chunks (List[str]): Potongan teks.
    
    Returns:
        Tuple[vectorizer, matrix]: TF-IDF vectorizer dan matriksnya.
    """
    if not chunks:
        logging.warning("Tidak ada chunks untuk diindeks.")
        return None, None
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(chunks)
    return vectorizer, tfidf_matrix

def analyze_chunks(chunks: List[str]):
    """
    Menganalisis statistik dasar dari chunks yang dibuat.
    
    Args:
        chunks (List[str]): Potongan teks hasil pemrosesan.
    """
    if not chunks:
        logging.info("Tidak ada chunks untuk dianalisis.")
        return
    
    chunk_lengths = [len(word_tokenize(chunk)) for chunk in chunks]
    if not chunk_lengths:
        logging.info("Tidak ada chunks untuk dianalisis.")
        return

    avg_length = sum(chunk_lengths) / len(chunks)
    
    logging.info("\n--- Analisis Chunks ---")
    logging.info(f"Total chunks: {len(chunks)}")
    logging.info(f"Rata-rata token per chunk: {avg_length:.2f}")
    logging.info("\n3 Sample Chunk Pertama:")
    for i, chunk in enumerate(chunks[:3]):
        logging.info(f"Chunk {i+1} (panjang {len(word_tokenize(chunk))} token):\n{chunk[:200]}...")

def main():
    """
    Fungsi utama untuk menjalankan pipeline pemrosesan PDF.
    Skrip ini sekarang menerima direktori berisi file PDF.
    """
    parser = argparse.ArgumentParser(description='Script untuk memproses dokumen PERDA dan membuat TF-IDF index.')
    parser.add_argument('pdf_dir', type=str, help='Path ke direktori yang berisi file PDF PERDA.')
    parser.add_argument('--output', type=str, default="data/perda_data.pkl", help='Lokasi file output pickle.')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Atur level logging.')
    args = parser.parse_args()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    if not os.path.isdir(args.pdf_dir):
        logging.error(f"Direktori tidak ditemukan: {args.pdf_dir}.")
        return

    logging.info(f"Memulai proses persiapan data dari direktori: {args.pdf_dir}...")
    
    all_chunks: List[str] = []
    
    pdf_files = [f for f in os.listdir(args.pdf_dir) if f.endswith('.pdf')]
    if not pdf_files:
        logging.error("Tidak ada file PDF yang ditemukan di direktori tersebut.")
        return
        
    logging.info(f"Ditemukan {len(pdf_files)} file PDF untuk diproses.")
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(args.pdf_dir, pdf_file)
        logging.info(f"Memproses file: {pdf_file}")
        
        # 1. Ekstrak teks dari PDF
        raw_text = extract_text_from_pdf(pdf_path)
        if raw_text is None:
            logging.warning(f"Melewatkan file {pdf_file} karena tidak dapat mengekstrak teks.")
            continue
        
        # 2. Praproses teks
        cleaned_text = preprocess_text(raw_text)
        
        # 3. Potong teks menjadi chunks
        document_chunks = chunk_text_by_structure(cleaned_text)
        if not document_chunks:
            logging.warning(f"Melewatkan file {pdf_file} karena gagal membagi dokumen menjadi chunks.")
            continue
            
        all_chunks.extend(document_chunks)
    
    if not all_chunks:
        logging.error("Tidak ada chunks yang dihasilkan dari semua dokumen. Proses dihentikan.")
        return

    # 4. Analisis statistik chunks
    analyze_chunks(all_chunks)
    
    # 5. Buat indeks TF-IDF dari semua chunks
    vectorizer, tfidf_matrix = create_tfidf_index(all_chunks)
    if vectorizer is None or tfidf_matrix is None:
        logging.error("Gagal membuat TF-IDF index.")
        return
        
    # 6. Simpan hasil ke dalam file pickle
    processed_data = {
        'chunks': all_chunks,
        'vectorizer': vectorizer,
        'tfidf_matrix': tfidf_matrix
    }
    
    joblib.dump(processed_data, args.output)
    logging.info(f"\nProses selesai. Data berhasil disimpan ke {args.output}")
    logging.info(f"Ukuran TF-IDF matrix: {tfidf_matrix.shape}")

if __name__ == "__main__":
    main()