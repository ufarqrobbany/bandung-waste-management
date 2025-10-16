import json

def update_ground_truth(generated_file, evaluation_file, output_file):
    """
    Memperbarui field 'ground_truth' di file jawaban yang dihasilkan
    dengan nilai dari file evaluasi berdasarkan pertanyaan yang cocok.

    Args:
        generated_file (str): Path ke file JSON dengan jawaban yang dihasilkan.
        evaluation_file (str): Path ke file JSON dengan data ground truth yang baru.
        output_file (str): Path untuk menyimpan file JSON yang telah diperbarui.
    """
    try:
        # Muat data evaluasi dan buat kamus pencarian
        # di mana kuncinya adalah pertanyaan dan nilainya adalah ground_truth.
        with open(evaluation_file, 'r', encoding='utf-8') as f:
            evaluation_data = json.load(f)
        ground_truth_map = {item['question']: item['ground_truth'] for item in evaluation_data}

        # Muat data jawaban yang dihasilkan
        with open(generated_file, 'r', encoding='utf-8') as f:
            generated_data = json.load(f)

        # Kunci dalam file spesifik ini adalah "Reranker (Seimbang)"
        # Anda dapat mengubah ini jika kuncinya berbeda di file lain.
        main_key = "Baseline (TF-IDF Saja)"
        if main_key not in generated_data:
            print(f"Error: Kunci '{main_key}' tidak ditemukan di {generated_file}")
            return

        # Iterasi melalui daftar pertanyaan dan perbarui ground_truth
        updated_count = 0
        for item in generated_data[main_key]:
            question = item.get('question')
            if question in ground_truth_map:
                item['ground_truth'] = ground_truth_map[question]
                updated_count += 1

        # Simpan data yang diperbarui ke file baru
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(generated_data, f, indent=4, ensure_ascii=False)

        print(f"Berhasil memperbarui {updated_count} entri.")
        print(f"Data yang diperbarui telah disimpan ke '{output_file}'.")

    except FileNotFoundError as e:
        print(f"Error: File {e.filename} tidak ditemukan.")
    except json.JSONDecodeError:
        print("Error: Gagal mendekode JSON dari salah satu file.")
    except Exception as e:
        print(f"Terjadi kesalahan tak terduga: {e}")

# Tentukan nama file
generated_answers_file = 'data/generated_answers_baseline_5.json'
evaluation_new_file = 'data/evaluation_new.json'
output_filename = 'data/updated_generated_answers_baseline_5.json'

# Jalankan fungsi
update_ground_truth(generated_answers_file, evaluation_new_file, output_filename)