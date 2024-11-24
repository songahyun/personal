import re
import random
import torch
import openai
import pandas as pd
from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForMaskedLM
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")

# 데이터 파일 로드
data_total = pd.read_excel("data_total.xlsx", dtype=str).dropna(subset=['신조어'])
data_ex_2 = pd.read_csv("data_ex_2.csv")

word_list_total = data_total['신조어'].tolist()
word_list_ex_2 = data_ex_2['신조어'].tolist()

word_replacements = dict(zip(data_total['신조어'], data_total['뜻']))
word_descriptions = dict(zip(data_total['신조어'], data_total['설명']))

# BERT 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertForMaskedLM.from_pretrained('bert-base-multilingual-cased')

# TF-IDF 설정
data_total['설명'] = data_total['설명'].fillna('').astype(str)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data_total['설명'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
word_to_index = dict(zip(data_total['신조어'], data_total.index))

def get_similar_words(word, cosine_sim=cosine_sim, threshold=0.145):
    if word not in word_to_index:
        return []

    idx = word_to_index[word]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [score for score in sim_scores if score[1] >= threshold and score[0] != idx]

    word_indices = [score[0] for score in sim_scores]
    similar_words = data_total['신조어'].iloc[word_indices]
    return similar_words.tolist()

def find_similar_words(sentence, word_list, threshold=65.0):
    found_words = []
    sentence_lower = sentence.lower()
    word_list_lower = [word.lower() for word in word_list]

    for word, word_lower in zip(word_list, word_list_lower):
        for i in range(len(sentence_lower) - len(word_lower) + 1):
            candidate = sentence_lower[i:i + len(word_lower)]
            similarity = fuzz.ratio(word_lower, candidate)
            if similarity >= threshold:
                found_words.append((word, sentence[i:i + len(word)], similarity))
    found_words = sorted(found_words, key=lambda x: -x[2])
    used = set()
    unique_words = []
    for word in found_words:
        if word[1] not in used:
            unique_words.append(word)
            used.add(word[1])
    return unique_words

def replace_with_bert(sentence, found_words, word_replacements, word_descriptions):
    replaced_sentence = sentence
    replacement_words = []
    descriptions = []

    for original_word, similar_word, _ in found_words:
        masked_sentence = sentence.replace(similar_word, tokenizer.mask_token)
        inputs = tokenizer(masked_sentence, return_tensors='pt')
        mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        mask_token_logits = logits[0, mask_token_index, :]
        top_token = torch.argmax(mask_token_logits, dim=1)
        predicted_token = tokenizer.decode(top_token)

        replaced_sentence = replaced_sentence.replace(similar_word, word_replacements[original_word], 1)
        if original_word not in replacement_words:
            replacement_words.append(original_word)
            descriptions.append(word_descriptions[original_word])

    return replaced_sentence, replacement_words, descriptions

def make_sentence_natural(sentence):
    prompt = f"{sentence}를 문법에 맞고 더 자연스럽게 수정해줘. 주어, 목적어, 말투는 그대로 해줘"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
        )
        natural_sentence = response['choices'][0]['message']['content'].strip()
        return natural_sentence
    except openai.OpenAIError:
        print("API 요청에 실패했습니다.")
        return sentence

def is_only_slang(sentence, word_list):
    tokens = sentence.split()
    return all(token in word_list for token in tokens)

def get_example_sentences(word, data):
    sources = ['네이버 블로그', '유튜브/X', '디시인사이드', '인스타그램', '나무위키']
    comments = data.loc[data['신조어'] == word, sources].dropna(axis=1).values.flatten()
    if comments.size > 0:
        return random.choice(comments)
    else:
        return "No example sentence available."

def generate_example_sentences(words, model_id):
    examples = {}
    for word in words:
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "This is a slang word example generator."},
                {"role": "user", "content": f"'{word}'이라는 신조어를 사용한 예문을 만들어주세요."}
            ]
        )
        examples[word] = response.choices[0].message['content'].strip()
    return examples

@app.route('/', methods=['GET', 'POST'])
def translate():
    translated_text = None
    replacement_words = []
    descriptions = []
    similar_words = {}
    example_sentences = {}
    sentence = ""

    if request.method == 'POST':
        sentence = request.form['input_text']
        sentence = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', sentence)
        found_words = find_similar_words(sentence, word_list_total)
        if not found_words:
            translated_text = "신조어가 쓰이지 않았거나, 업데이트 되지 않은 신조어입니다."
        else:
            replaced_sentence, replacement_words, descriptions = replace_with_bert(sentence, found_words, word_replacements, word_descriptions)
            if not is_only_slang(sentence, word_list_total):
                replaced_sentence = make_sentence_natural(replaced_sentence)
            translated_text = replaced_sentence

            for word in replacement_words:
                if word in word_list_ex_2:
                    # Fine-tuning 작업 ID
                    fine_tuning_job_id = os.getenv("FINE_TUNING_JOB_ID")


                    # Fine-tuning 작업 상태 확인
                    fine_tuning_job = openai.FineTuningJob.retrieve(fine_tuning_job_id)

                    # Fine-tuned 모델 ID 확인
                    fine_tuned_model_id = fine_tuning_job['fine_tuned_model']

                    # 신조어 예문 생성
                    example_sentences[word] = generate_example_sentences([word], fine_tuned_model_id).get(word, "No example sentence available.")
                else:
                    example_sentences[word] = get_example_sentences(word, data_total)
                similar_words[word] = get_similar_words(word)

    return render_template("trans5.html", input_text=sentence, translated_text=translated_text, replacement_words=replacement_words, descriptions=descriptions, similar_words=similar_words, example_sentences=example_sentences, zip=zip)

if __name__ == '__main__':
    app.run(debug=False)