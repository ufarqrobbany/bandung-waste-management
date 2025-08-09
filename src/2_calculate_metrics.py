import json
import pandas as pd
from rouge_score import rouge_scorer
from collections import defaultdict

INPUT_FILE = "data/generated_answers.json"

def calculate_f1(generated_text, reference_text):
    """Menghitung F1 score berdasarkan tumpang tindih token."""
    gen_tokens = set(generated_text.lower().split())
    ref_tokens = set(reference_text.lower().split())
    if not ref_tokens: return 0.0
    common_tokens = gen_tokens.intersection(ref_tokens)
    if not common_tokens: return 0.0
    precision = len(common_tokens) / len(gen_tokens)
    recall = len(common_tokens) / len(ref_tokens)
    if precision + recall == 0: return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_all_metrics():
    """
    Membaca jawaban yang telah disimpan dan menghitung metrik evaluasi.
    """
    print(f"Memuat jawaban yang dihasilkan dari: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_generated_answers = json.load(f)
    except FileNotFoundError:
        print(f"File '{INPUT_FILE}' tidak ditemukan. Jalankan '1_generate_answers.py' terlebih dahulu.")
        return

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []

    print("\nMemulai perhitungan metrik...")

    # Loop melalui setiap mode yang ada di file
    for mode_name, answers in all_generated_answers.items():
        print(f"--- Menghitung metrik untuk Mode: {mode_name} ---")
        
        total_scores = defaultdict(float)
        num_items = len(answers)

        if num_items == 0:
            continue

        # Loop melalui setiap jawaban yang dihasilkan untuk mode ini
        for item in answers:
            generated_answer = item['generated_answer']
            ground_truth = item['ground_truth']

            # Hitung skor
            rouge_scores = scorer.score(ground_truth, generated_answer)
            f1 = calculate_f1(generated_answer, ground_truth)

            # Akumulasi skor
            total_scores['f1'] += f1
            total_scores['rouge1'] += rouge_scores['rouge1'].fmeasure
            total_scores['rouge2'] += rouge_scores['rouge2'].fmeasure
            total_scores['rougeL'] += rouge_scores['rougeL'].fmeasure

        # Hitung rata-rata skor
        avg_scores = {key: value / num_items for key, value in total_scores.items()}

        results.append({
            'Mode': mode_name,
            'Avg F1': avg_scores['f1'],
            'Avg ROUGE-1': avg_scores['rouge1'],
            'Avg ROUGE-2': avg_scores['rouge2'],
            'Avg ROUGE-L': avg_scores['rougeL']
        })

    # Tampilkan hasil dalam tabel pandas
    if not results:
        print("Tidak ada hasil untuk ditampilkan.")
        return
        
    results_df = pd.DataFrame(results)
    print("\n\n--- HASIL EVALUASI AKHIR ---")
    print(results_df.to_string())


if __name__ == "__main__":
    calculate_all_metrics()