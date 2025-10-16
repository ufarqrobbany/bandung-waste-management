import json
import pandas as pd
from collections import defaultdict

# Pustaka untuk metrik peringkat
import pytrec_eval

# Pustaka untuk metrik kesamaan semantik dan uji statistik
from bert_score import score as bert_score_func
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from scipy import stats

# Nama file input dan output
INPUT_FILE = "data/new_generated_answers_baseline_5.json"
OUTPUT_STAT_FILE = "data/new_generated_answers_baseline_5_stats.json"

def calculate_ranking_metrics(answers):
    """
    Menghitung metrik peringkat (MRR, nDCG) menggunakan pytrec_eval.
    """
    qrels = {}
    run = {}

    for i, item in enumerate(answers):
        query_id = str(i)
        retrieved_chunks = item.get("retrieved_chunks_with_scores", [])
        ground_truth_text = item.get("ground_truth")

        if not ground_truth_text or not retrieved_chunks:
            continue

        qrels[query_id] = {}
        for chunk_text, _ in retrieved_chunks:
            # Asumsi: Jika ground truth ada di dalam chunk, chunk itu relevan.
            # Ini bisa diganti dengan metode yang lebih canggih.
            if ground_truth_text.lower() in chunk_text.lower():
                qrels[query_id][chunk_text] = 1
                break

        if not qrels[query_id]:
            del qrels[query_id]
            continue
            
        run[query_id] = {chunk_text: score for chunk_text, score in retrieved_chunks}

    if not qrels or not run:
        return 0.0, 0.0, [0.0] * len(answers), [0.0] * len(answers)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'recip_rank', 'ndcg'})
    results = evaluator.evaluate(run)
    
    avg_mrr = pytrec_eval.compute_aggregated_metric(results, 'recip_rank')
    avg_ndcg = pytrec_eval.compute_aggregated_metric(results, 'ndcg')

    # Dapatkan skor per-pertanyaan
    mrr_scores = [results.get(str(i), {}).get('recip_rank', 0) for i in range(len(answers))]
    ndcg_scores = [results.get(str(i), {}).get('ndcg', 0) for i in range(len(answers))]

    return avg_mrr, avg_ndcg, mrr_scores, ndcg_scores

def calculate_all_metrics():
    """
    Menghitung semua metrik dan melakukan uji statistik.
    """
    print(f"Memuat jawaban yang dihasilkan dari: {INPUT_FILE}")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            all_generated_answers = json.load(f)
    except FileNotFoundError:
        print(f"File '{INPUT_FILE}' tidak ditemukan. Jalankan '1_generate_answers.py' terlebih dahulu.")
        return

    print("Memuat model sentence-transformer untuk Cosine Similarity...")
    try:
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    except Exception as e:
        print(f"Gagal memuat model sentence-transformer: {e}")
        return

    results = []
    all_scores = defaultdict(dict) # Menyimpan semua skor per-pertanyaan

    print("\nMemulai perhitungan metrik...")

    for mode_name, answers in all_generated_answers.items():
        print(f"--- Menghitung metrik untuk Mode: {mode_name} ---")
        
        num_items = len(answers)
        if num_items == 0:
            continue
            
        generated_texts = [item['generated_answer'] for item in answers]
        ground_truths = [item['ground_truth'] for item in answers]

        # 1. Hitung metrik peringkat (MRR, nDCG)
        print("Menghitung MRR dan nDCG...")
        try:
            if "retrieved_chunks_with_scores" not in answers[0]:
                print("Peringatan: Data peringkat tidak ditemukan di file input.")
                avg_mrr, avg_ndcg, mrr_scores, ndcg_scores = 0.0, 0.0, [0.0]*num_items, [0.0]*num_items
            else:
                avg_mrr, avg_ndcg, mrr_scores, ndcg_scores = calculate_ranking_metrics(answers)
        except Exception as e:
            print(f"Gagal menghitung metrik peringkat: {e}")
            avg_mrr, avg_ndcg, mrr_scores, ndcg_scores = 0.0, 0.0, [0.0]*num_items, [0.0]*num_items
        
        all_scores[mode_name]['mrr'] = mrr_scores
        all_scores[mode_name]['ndcg'] = ndcg_scores

        # 2. Hitung BERTScore
        print("Menghitung BERTScore...")
        try:
            _, _, bert_f1_scores = bert_score_func(generated_texts, ground_truths, lang="id", verbose=False)
            avg_bert_f1 = bert_f1_scores.mean().item()
            bert_f1_scores = [s.item() for s in bert_f1_scores]
        except Exception as e:
            print(f"Gagal menghitung BERTScore: {e}")
            avg_bert_f1 = 0.0
            bert_f1_scores = [0.0] * num_items

        all_scores[mode_name]['bert_f1'] = bert_f1_scores

        # 3. Hitung Cosine Similarity
        print("Menghitung Cosine Similarity...")
        try:
            generated_embeddings = model.encode(generated_texts, convert_to_tensor=False)
            ground_truth_embeddings = model.encode(ground_truths, convert_to_tensor=False)
            cosine_scores = [cosine_similarity(gen_emb.reshape(1, -1), gt_emb.reshape(1, -1))[0][0] for gen_emb, gt_emb in zip(generated_embeddings, ground_truth_embeddings)]
            avg_cosine_similarity = sum(cosine_scores) / len(cosine_scores) if cosine_scores else 0.0
        except Exception as e:
            print(f"Gagal menghitung Cosine Similarity: {e}")
            avg_cosine_similarity = 0.0
            cosine_scores = [0.0] * num_items
        
        all_scores[mode_name]['cosine_similarity'] = cosine_scores

        results.append({
            'Mode': mode_name,
            'Avg MRR': avg_mrr,
            'Avg nDCG': avg_ndcg,
            'Avg Cosine Similarity': avg_cosine_similarity,
            'Avg BERTScore F1': avg_bert_f1,
        })
    
    # Lakukan Uji Statistik
    print("\n--- Melakukan Uji Statistik (Wilcoxon Signed-Rank Test) ---")
    if len(all_scores) == 2:
        modes = list(all_scores.keys())
        baseline_mode = modes[0]
        reranker_mode = modes[1]

        stat_results = {
            'Test': 'Wilcoxon Signed-Rank Test',
            'Comparing': f'{reranker_mode} vs {baseline_mode}',
            'p_values': {}
        }
        
        for metric in ['mrr', 'ndcg', 'bert_f1', 'cosine_similarity']:
            try:
                stat, p_value = stats.wilcoxon(all_scores[baseline_mode][metric], all_scores[reranker_mode][metric])
                stat_results['p_values'][metric] = p_value
                print(f"Metrik: {metric.upper()}")
                print(f"  p-value: {p_value:.5f} ({'Signifikan' if p_value < 0.05 else 'Tidak Signifikan'})")
            except ValueError:
                print(f"Metrik: {metric.upper()} - Tidak bisa melakukan uji (skor terlalu mirip atau kurang data).")
        
        with open(OUTPUT_STAT_FILE, 'w', encoding='utf-8') as f:
            json.dump(stat_results, f, indent=4)
        print(f"\nHasil uji statistik disimpan di: {OUTPUT_STAT_FILE}")

    # Tampilkan hasil dalam tabel pandas
    if not results:
        print("\nTidak ada hasil untuk ditampilkan.")
        return
        
    results_df = pd.DataFrame(results)
    print("\n\n--- HASIL EVALUASI AKHIR ---")
    print(results_df.to_string())

if __name__ == "__main__":
    calculate_all_metrics()