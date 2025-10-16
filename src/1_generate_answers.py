import json
import logging
import time
from retriever import DocumentRetriever
from generator import LLMGeneratorSync as LLMGenerator

# Konfigurasi logging agar tidak terlalu ramai
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# Kamus konfigurasi retriever (sama seperti sebelumnya)
RETRIEVER_MODES = {
    # "Baseline (TF-IDF Saja)": {
    #     "use_reranker": False, "top_k": 5, "initial_k": 5
    # },
    "Reranker (Seimbang)": {
        "use_reranker": True, "top_k": 5, "initial_k": 50
    },
    # "Reranker (Akurasi Tinggi)": {
    #     "use_reranker": True, "top_k": 5, "initial_k": 200
    # },
    # "Reranker (Akurasi Sangat Tinggi)": {
    #     "use_reranker": True, "top_k": 5, "initial_k": 500
    # },
    # "Reranker (Cepat)": {
    #     "use_reranker": True, "top_k": 3, "initial_k": 20
    # }
}

OUTPUT_FILE = "data/new_generated_answers_reranker_5_50.json"

def generate_all_answers():
    """
    Menghasilkan jawaban dari semua mode untuk semua pertanyaan evaluasi
    dan menyimpannya ke dalam satu file JSON.
    """
    print("Memuat komponen (Retriever dan Generator)...")
    try:
        retriever = DocumentRetriever()
        generator = LLMGenerator()
    except Exception as e:
        print(f"Gagal memuat komponen: {e}")
        return

    print("Memuat dataset evaluasi...")
    try:
        with open('data/new_evaluation.json', 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)
    except FileNotFoundError:
        print("File 'data/new_evaluation.json' tidak ditemukan.")
        return

    all_generated_answers = {}

    print(f"\nMemulai proses pembuatan jawaban untuk {len(RETRIEVER_MODES)} mode...")

    # Loop melalui setiap mode konfigurasi
    for mode_name, config in RETRIEVER_MODES.items():
        print(f"\n--- Memproses Mode: {mode_name} ---")
        
        mode_answers = []
        
        # Loop melalui setiap item di dataset evaluasi
        for i, item in enumerate(evaluation_data):
            question = item['question']
            
            print(f"  > Menghasilkan jawaban untuk pertanyaan {i+1}/{len(evaluation_data)}...", end="", flush=True)
            
            # 1. Retrieve chunks
            retrieved_chunks_with_scores = retriever.retrieve_chunks(
                query=question,
                top_k=config['top_k'],
                initial_k=config['initial_k'],
                use_reranker=config['use_reranker']
            )
            # retrieved_chunks = [c[0] for c in retrieved_chunks_with_scores]

            sanitized_chunks_with_scores = [
                (chunk, float(score)) for chunk, score in retrieved_chunks_with_scores
            ]

            # Use the sanitized chunks for generation
            retrieved_chunks = [c[0] for c in sanitized_chunks_with_scores]


            # 2. Generate answer
            generated_answer = generator.generate_answer(question, retrieved_chunks)

            mode_answers.append({
                "question": question,
                "generated_answer": generated_answer,
                "ground_truth": item["ground_truth"], # Sertakan ground truth untuk kemudahan
                "retrieved_chunks_with_scores": sanitized_chunks_with_scores # Use the sanitized list
            })
            
            print(" Selesai.")
            
            # Jeda waktu konservatif untuk menghindari rate limit
            time.sleep(6)

        all_generated_answers[mode_name] = mode_answers

    # Simpan semua hasil ke file JSON
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_generated_answers, f, indent=4, ensure_ascii=False)
        print(f"\n\nProses selesai. Semua jawaban telah disimpan di: {OUTPUT_FILE}")
    except Exception as e:
        print(f"Gagal menyimpan file output: {e}")


if __name__ == "__main__":
    generate_all_answers()