import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save_classifier(train_path, validation_path, model_path="classifier_model.pkl"):
    """
    Melatih model klasifikasi teks berbasis Multinomial Naive Bayes dan menyimpannya ke file.

    Dataset diharapkan memiliki dua kolom: satu untuk teks (e.g., 'Phrase') dan satu untuk label (e.g., 'Class').
    Data pelatihan dan validasi akan digabung, lalu digunakan untuk melatih pipeline klasifikasi.

    Args:
        train_path (str): Path ke file CSV data pelatihan.
        validation_path (str): Path ke file CSV data validasi.
        model_path (str): Path tempat menyimpan model hasil pelatihan (.pkl).
    
    Returns:
        None. Model disimpan ke disk jika pelatihan berhasil.
    """
    try:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(validation_path)
        full_df = pd.concat([train_df, val_df], ignore_index=True)

        logging.info(f"Dataset dimuat. Total sampel: {len(full_df)}")
        
        # Tampilkan informasi data
        logging.info("Inspeksi data:")
        logging.info(full_df.head())
        logging.info(f"Nama kolom: {full_df.columns.tolist()}")

        # Ubah nama kolom sesuai kebutuhan
        text_column_name = 'Phrase'
        label_column_name = 'Class'

        if text_column_name not in full_df.columns:
            logging.error(f"Kolom '{text_column_name}' tidak ditemukan di dataset.")
            return
        if label_column_name not in full_df.columns:
            logging.error(f"Kolom '{label_column_name}' tidak ditemukan di dataset.")
            return

        # Pipeline: TF-IDF + Naive Bayes
        text_classifier = Pipeline([
            ('vectorizer', TfidfVectorizer()),
            ('classifier', MultinomialNB())
        ])
        
        logging.info("Memulai pelatihan model klasifikasi...")
        text_classifier.fit(full_df[text_column_name], full_df[label_column_name])
        logging.info("Pelatihan selesai.")

        # Simpan model
        joblib.dump(text_classifier, model_path)
        logging.info(f"Model disimpan ke: {model_path}")

    except Exception as e:
        logging.error(f"Terjadi kesalahan saat pelatihan: {e}")

if __name__ == "__main__":
    train_and_save_classifier(
        train_path="data/raw/train.csv",
        validation_path="data/raw/validation.csv",
        model_path="data/classifier_model.pkl"
    )