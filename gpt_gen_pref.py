import os
import json
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

import random
import openai
import time


openai.api_key = "your key"


def call_chatcompletion_with_retry(*args, **kwargs):
    max_retries = 5
    base_sleep = 10

    for attempt in range(max_retries):
        try:
            return openai.ChatCompletion.create(*args, **kwargs)
        except (openai.error.APIConnectionError,
                openai.error.RateLimitError,
                openai.error.Timeout,
                openai.error.APIError) as e:
            if attempt == max_retries - 1:
                raise e
            sleep_time = base_sleep * (2 ** attempt) + random.random()
            print(f"[{e.__class__.__name__}] {e}")
            print(f" -> Retry after {sleep_time:.1f}second. (attempt {attempt+1}/{max_retries})")
            time.sleep(sleep_time)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


labse_model = AutoModel.from_pretrained("sentence-transformers/LaBSE").to(device)
labse_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
labse_model.eval()


TARGET_LANGUAGES = ["en", "ko", "zh", "fr", "ja", "it", "pt", "es"]


def generate_multilingual_answers_openai_en(qid, question, top5_passages):
    system_message = (
        "You are a highly capable multilingual assistant. "
        "Here are some reference documents:\n\n"
        f"{top5_passages}\n\n"
        "The user wants answers in multiple languages. "
        "Please follow these rules strictly:\n\n"
        "1) Return your final answer as a valid JSON object.\n"
        "2) The JSON object must contain exactly these keys: "
        f"{', '.join(TARGET_LANGUAGES)}.\n"
        "3) Each field's value must be the answer written in that respective language.\n"
        "4) Do not include any additional text outside the JSON (e.g., no Markdown or explanations).\n"
        "5) Ensure it is valid JSON with correct format.\n"
    )

    user_message = (
        f"Question: {question}\n\n"
        f"Please provide the answers in JSON form for each of the following languages: {TARGET_LANGUAGES}."
    )
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message},
    ]

    response = call_chatcompletion_with_retry(
        model="gpt-4o-mini",  
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )

    content = response.choices[0].message.content.strip()

    try:
        answers = json.loads(content)
    except json.JSONDecodeError:
        answers = {}

    for lang in TARGET_LANGUAGES:
        if lang not in answers:
            answers[lang] = ""
        if not isinstance(answers[lang], str):
            answers[lang] = str(answers[lang])

    return answers


def generate_multilingual_answers_openai_zh(qid, question, top5_passages):
    system_message = (
        "你是一位多语言的高水平AI助手，可以参考以下文档：\n\n"
        f"{top5_passages}\n\n"
        "用户想要多种语言的答案。请严格遵守以下规则：\n\n"
        "1) 最终答案必须以有效的JSON对象形式返回。\n"
        "2) 该JSON对象必须包含以下键："
        f"{', '.join(TARGET_LANGUAGES)}。\n"
        "3) 每个字段对应的值应使用该语言撰写。\n"
        "4) 不要在JSON之外添加任何额外文本（例如Markdown或解释）。\n"
        "5) 确保输出的是符合格式的JSON。\n"
    )

    user_message = (
        f"问题：{question}\n\n"
        f"请以JSON形式为以下语言分别提供答案：{TARGET_LANGUAGES}。"
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message},
    ]

    response = call_chatcompletion_with_retry(
        model="gpt-4o-mini",  
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )

    content = response.choices[0].message.content.strip()

    try:
        answers = json.loads(content)
    except json.JSONDecodeError:
        answers = {}

    for lang in TARGET_LANGUAGES:
        if lang not in answers:
            answers[lang] = ""
        if not isinstance(answers[lang], str):
            answers[lang] = str(answers[lang])

    return answers


def generate_multilingual_answers_openai_ko(qid, question, top5_passages):
    system_message = (
        "당신은 다양한 언어를 다룰 수 있는 능숙한 AI 어시스턴트입니다. "
        "다음은 참조할 수 있는 문서들입니다:\n\n"
        f"{top5_passages}\n\n"
        "사용자는 여러 언어로 된 답변을 원합니다. "
        "다음 규칙을 엄격히 따라주세요:\n\n"
        "1) 최종 답변을 유효한 JSON 객체로 반환해야 합니다.\n"
        "2) JSON 객체에는 다음 키가 정확히 포함되어야 합니다: "
        f"{', '.join(TARGET_LANGUAGES)}.\n"
        "3) 각 필드의 값은 해당 언어로 작성된 답변이어야 합니다.\n"
        "4) JSON 외에 추가 텍스트(예: 마크다운, 설명)는 넣지 마세요.\n"
        "5) 반드시 올바른 JSON 형식을 유지해야 합니다.\n"
    )

    user_message = (
        f"질문: {question}\n\n"
        f"다음 언어들에 대해 각각 답변을 JSON 형태로 제공해주세요: {TARGET_LANGUAGES}."
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user",   "content": user_message},
    ]

    response = call_chatcompletion_with_retry(
        model="gpt-4o-mini",  
        messages=messages,
        temperature=0.7,
        max_tokens=512
    )

    content = response.choices[0].message.content.strip()

    try:
        answers = json.loads(content)
    except json.JSONDecodeError:
        answers = {}

    for lang in TARGET_LANGUAGES:
        if lang not in answers:
            answers[lang] = ""
        if not isinstance(answers[lang], str):
            answers[lang] = str(answers[lang])

    return answers


def get_labse_embeddings(texts):
    texts = [str(t) if t is not None else "" for t in texts]

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


def process_dataset(dataset_path, top5_content_path, generate_func, output_file):
    dataset = load_from_disk(dataset_path)

    with open(top5_content_path, encoding='utf-8') as f:
        top5_data = json.load(f)
    
    results = []
    
    for example in tqdm(dataset, desc=f"Processing {dataset_path}"):
        qid = str(example["id"])
        question = example["content"]
        
        passages_for_qid = top5_data.get(qid, [])[:5]  
        top5_passages = "\n\n".join([doc["content"] for doc in passages_for_qid])

        answers = generate_func(qid, question, top5_passages)
        
        texts = list(answers.values())  
        embeddings = get_labse_embeddings(texts)
        similarity_matrix = cosine_similarity(embeddings)
        
        results.append({
            "qid": qid,
            "question": question,
            "answers": answers,
            "similarity_matrix": similarity_matrix.tolist()
        })
        
        time.sleep(1)

    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"=> Saved results to {output_file}")


if __name__ == "__main__":
    base_path = "datasets"

    en_dataset_path = os.path.join(base_path, "mkqa_en_train")
    en_top5_content_path = "en_top5_with_content.json"
    en_output_file = "gpt_multilingual_analysis_results_en.json"
    process_dataset(
        dataset_path=en_dataset_path,
        top5_content_path=en_top5_content_path,
        generate_func=generate_multilingual_answers_openai_en, 
        output_file=en_output_file
    )

    zh_dataset_path = os.path.join(base_path, "mkqa_zh_cn_train")
    zh_top5_content_path = "zh_top5_with_content.json"
    zh_output_file = "gpt_multilingual_analysis_results_zh.json"
    process_dataset(
        dataset_path=zh_dataset_path,
        top5_content_path=zh_top5_content_path,
        generate_func=generate_multilingual_answers_openai_zh,  
        output_file=zh_output_file
    )

    ko_dataset_path = os.path.join(base_path, "mkqa_ko_train")
    ko_top5_content_path = "ko_top5_with_content.json"
    ko_output_file = "gpt_multilingual_analysis_results_ko.json"
    process_dataset(
        dataset_path=ko_dataset_path,
        top5_content_path=ko_top5_content_path,
        generate_func=generate_multilingual_answers_openai_ko,  
        output_file=ko_output_file
    )

    print("All tasks are done (EN → ZH → KO).")
