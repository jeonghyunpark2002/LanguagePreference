import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

plt.rcParams.update({
    'font.size': 30,          
    'axes.titlesize': 20,    
    'axes.labelsize': 18,    
    'xtick.labelsize': 40,    
    'ytick.labelsize': 40,    
    'figure.dpi': 300,       
    'savefig.dpi': 300,       
    'pdf.fonttype': 42,       
    'ps.fonttype': 42
})

def compute_average_similarity_matrix(json_file_path):
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file {json_file_path}: {e}")
        return None

    matrices = []
    for item in data:
        if "similarity_matrix" in item:
            matrices.append(np.array(item["similarity_matrix"]))
    
    if not matrices:
        print(f"No similarity matrices found in {json_file_path}.")
        return None
    
    average_matrix = np.mean(matrices, axis=0)
    return average_matrix

def compute_average_per_language(average_matrix):
    languages = ["en", "ko", "zh", "fr", "ja", "it", "pt", "es"]
    n = average_matrix.shape[0]
    avg_per_lang = {}
    
    for i in range(n):
        row_sum = np.sum(average_matrix[i]) - average_matrix[i, i]
        avg_value = row_sum / (n - 1)
        
        lang = languages[i] if i < len(languages) else f"lang_{i}"
        avg_per_lang[lang] = avg_value
    
    return avg_per_lang

def compute_generator_language_preference(file_list):
    combined_pref = {}
    count = 0
    for file in file_list:
        matrix = compute_average_similarity_matrix(file)
        if matrix is None:
            continue
        pref = compute_average_per_language(matrix)
        if not combined_pref:
            combined_pref = {lang: pref[lang] for lang in pref}
        else:
            for lang in combined_pref:
                combined_pref[lang] += pref[lang]
        count += 1
    if count > 0:
        for lang in combined_pref:
            combined_pref[lang] /= count
    else:
        print(f"No valid files found in {file_list}")
    return combined_pref

def plot_generator_language_preferences(generator_preferences, output_path="generator_language_preference.pdf"):
    languages = ["en", "ko", "zh", "fr", "ja", "it", "pt", "es"]
    plt.figure(figsize=(16, 12))
    
    palette = sns.color_palette("viridis", len(generator_preferences))
    
    for idx, (generator, pref_dict) in enumerate(generator_preferences.items()):
        y_values = [pref_dict.get(lang, np.nan) for lang in languages]
        plt.plot(languages, y_values, marker='o', markersize=10, linewidth=3,
                 label=generator, color=palette[idx])
    
    avg_values = []
    for lang in languages:
        values = [pref_dict.get(lang, np.nan) for pref_dict in generator_preferences.values()]
        avg_values.append(np.nanmean(values))
    
    plt.plot(languages, avg_values, marker='o', markersize=10, linewidth=3,
             linestyle='--', color='red')
    
    plt.title("Generator Language Preference (avg)", fontsize=40)
    
    plt.ylim(0.68, 1.00)
    
    ax = plt.gca()
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
    
    ax.tick_params(axis='both', which='both', length=0)
    
    
    plt.legend(fontsize=28)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"[Info] Language Preference line graph saved to '{output_path}'.")

def main():
    generator_files = {
        "llama": [
            "llama_multilingual_analysis_results_en.json", 
            #"llama_multilingual_analysis_results_zh.json", 
            #"llama_multilingual_analysis_results_ko.json"
        ],
        "aya": [
            "aya_multilingual_analysis_results_en.json", 
            #"aya_multilingual_analysis_results_zh.json", 
            #"aya_multilingual_analysis_results_ko.json"
        ],
        "gpt": [
            "gpt_multilingual_analysis_results_en.json", 
            #"gpt_multilingual_analysis_results_ko.json", 
            #"gpt_multilingual_analysis_results_zh.json"
        ]
    }
    
    generator_preferences = {}
    for generator, files in generator_files.items():
        pref = compute_generator_language_preference(files)
        if pref:
            generator_preferences[generator] = pref
            print(f"\n[{generator}] Average Similarity per Language:")
            for lang, value in pref.items():
                print(f"  {lang}: {value:.4f}")
        else:
            print(f"[Error] No data about {generator}.")
    
    plot_generator_language_preferences(generator_preferences, output_path="en_generator_language_preference.pdf")

if __name__ == "__main__":
    main()






