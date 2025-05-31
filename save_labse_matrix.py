import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams.update({
    'font.size': 18,          
    'axes.titlesize': 20,     
    'axes.labelsize': 18,     
    'xtick.labelsize': 28,    
    'ytick.labelsize': 28,    
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
        print("No similarity matrices found in the JSON file.")
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


def plot_similarity_matrix(average_matrix, output_path="similarity_matrix_ko.pdf"):
    languages = ["en", "ko", "zh", "fr", "ja", "it", "pt", "es"]
    
    plt.figure(figsize=(16, 12))  
    
    sns.heatmap(
        average_matrix, 
        annot=True, 
        fmt=".4f", 
        cmap="viridis",
        xticklabels=languages, 
        yticklabels=languages, 
        annot_kws={"size": 14}  
    )
    
    plt.title("Cross-Lingual Similarity Matrix (ko)", fontsize=22)
    
    plt.tight_layout()
    plt.savefig(
        output_path, 
        bbox_inches='tight',  
        dpi=300,              
        transparent=False     
    )
    plt.close()
    print(f"[Info] Similarity matrix saved to '{output_path}'.")

def plot_language_preferences(average_per_lang, output_path="language_preference_ko.pdf"):
    languages = list(average_per_lang.keys())
    lang_values = list(average_per_lang.values())
    
    plt.figure(figsize=(14, 10))  
    
    sns.barplot(
        x=languages, 
        y=lang_values, 
        palette="viridis",
        linewidth=2.5,      
    )
    
    plt.title("Generator Language Preference (ko)", fontsize=30)
    
    min_val = min(lang_values) - 0.05
    max_val = max(lang_values) + 0.05
    plt.ylim(max(0, min_val), min(1, max_val))
    
    plt.tight_layout()
    plt.savefig(
        output_path, 
        bbox_inches='tight', 
        dpi=300,
        transparent=False
    )
    plt.close()
    print(f"[Info] Average Similarity per Language saved to '{output_path}'.")

def main():
    json_file_path = "gpt_multilingual_analysis_results_ko.json"
    average_matrix = compute_average_similarity_matrix(json_file_path)
    
    if average_matrix is not None:
        print("Average Similarity Matrix (8x8):")
        print(average_matrix)
        
        average_per_lang = compute_average_per_language(average_matrix)
        print("\nAverage Similarity per Language (Self-Similarity 제외):")
        for lang, avg in average_per_lang.items():
            print(f"{lang}: {avg:.4f}")
        
        plot_similarity_matrix(average_matrix, output_path="gpt_similarity_matrix_ko.pdf")
        plot_language_preferences(average_per_lang, output_path="gpt_language_preference_ko.pdf")
        
if __name__ == "__main__":
    main()





