import os
import json
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoModel
)
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE").to(device)
labse_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
labse_model.eval()


aya_tokenizer = AutoTokenizer.from_pretrained(
    "CohereForAI/aya-expanse-8b",
    padding_side="left"
)
aya_tokenizer.pad_token = aya_tokenizer.eos_token

aya_model = AutoModelForCausalLM.from_pretrained(
    "CohereForAI/aya-expanse-8b",
    torch_dtype=torch.float16,
    device_map="auto" 
).eval()

aya_pipeline = pipeline(
    "text-generation",
    model=aya_model,
    tokenizer=aya_tokenizer,
    batch_size=16  
)

TARGET_LANGUAGES = ["en", "ko", "zh", "fr", "ja", "it", "pt", "es"]

def generate_multilingual_answers(qid, question, top5_passages):
    answers = {}
    
    prompts = []
    for lang in TARGET_LANGUAGES:
        system_msg = f"<system>\nYou are a multilingual assistant. Answer based on these documents:\n{top5_passages}</system>"
        user_msg = f"<user>\n{question}\nPlease respond in {lang}.</user>"
        prompts.append(system_msg + "\n\n" + user_msg)
    
    responses = aya_pipeline(
        prompts,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        pad_token_id=aya_tokenizer.eos_token_id,  
        return_full_text=False  
    )
    
    for idx, lang in enumerate(TARGET_LANGUAGES):
        answers[lang] = responses[idx][0]['generated_text'].strip()
    
    return answers

def get_labse_embeddings(texts):
    inputs = labse_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)
    with torch.no_grad():
        outputs = labse_model(**inputs)
    return outputs.pooler_output.cpu()

def process_dataset(dataset_path, top5_content_path):
    dataset = load_from_disk(dataset_path)
    with open(top5_content_path, encoding='utf-8') as f:
        top5_data = json.load(f)
    
    results = []
    
    for example in tqdm(dataset, desc="Processing queries"):
        qid = str(example["id"])
        question = example["content"]
        
        top5_passages = "\n\n".join([doc["content"] for doc in top5_data.get(qid, [])[:5]])
        answers = generate_multilingual_answers(qid, question, top5_passages)
        
        texts = list(answers.values())
        embeddings = get_labse_embeddings(texts)
        similarity_matrix = cosine_similarity(embeddings)
        
        results.append({
            "qid": qid,
            "question": question,
            "answers": answers,
            "similarity_matrix": similarity_matrix.tolist()
        })
    
    return results

if __name__ == "__main__":
    base_path = "datasets"
    
    mkqa_path = os.path.join(base_path, "mkqa_en_train")
    top5_content_path = "en_top5_with_content.json"
    
    analysis_results = process_dataset(mkqa_path, top5_content_path)
    
    with open("multilingual_analysis_results.json", "w", encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)