import json
from collections import defaultdict

input_file = "run.retrieve.top_50.mkqa_en.wiki-100w-multilingual-alllangs.dev.BAAI_bge-m3.trec"
output_file = "en_initial_rank.json"

query_results = defaultdict(list)

with open(input_file, 'r', encoding='utf-8') as infile:
    for line in infile:
        parts = line.strip().split()
        if len(parts) >= 4:
            query_id = parts[0]
            doc_id = parts[2]
            rank = int(parts[3])
            
            query_results[query_id].append({"doc_id": doc_id, "rank": rank})

with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(query_results, outfile, ensure_ascii=False, indent=4)

print(f"Filtered results saved to {output_file}")


