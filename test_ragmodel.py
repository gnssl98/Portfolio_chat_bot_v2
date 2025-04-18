import os
import sys
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    AutoModelForSequenceClassification
)
import torch
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import re
import glob
import fitz  # PyMuPDF
import PyPDF2  # 백업용
import json
from typing import List, Tuple, Dict
import nltk
from nltk.tokenize import sent_tokenize
from peft import PeftModel, PeftConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 환경 변수 로드
load_dotenv()

# 경로 설정
MODEL_ID = r"D:\dataset\fine_tuned_model\flan-t5-large"  # 로컬 기본 모델 경로
CACHE_DIR = r"D:\dataset\huggingface_cache"
VECTOR_DB_PATH = "./vector_db"
PORTFOLIO_DATA_DIR = "./portfolio_data"
TRANSLATION_MODEL_PATH = r"D:\dataset\fine_tuned_model\translation_model_finetunned"  # 파인튜닝된 번역 모델 경로

# Re-ranking 모델 설정
RERANKER_MODEL = "BAAI/bge-reranker-base"

# 모델 경로 설정
SAVE_DIR_KO_EN = r"D:\dataset\fine_tuned_model\translation_model_finetunned_ko_en"    # 한국어->영어 모델
SAVE_DIR_EN_KO = r"D:\dataset\fine_tuned_model\translation_model_finetunned_en_ko"    # 영어->한국어 모델

# 모델 로드
print("모델 로딩 중...")
qa_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
qa_tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
ko_en_model = MBartForConditionalGeneration.from_pretrained(SAVE_DIR_KO_EN)
ko_en_tokenizer = MBart50TokenizerFast.from_pretrained(SAVE_DIR_KO_EN)
en_ko_model = MBartForConditionalGeneration.from_pretrained(SAVE_DIR_EN_KO)
en_ko_tokenizer = MBart50TokenizerFast.from_pretrained(SAVE_DIR_EN_KO)

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 모델을 GPU로 이동
qa_model = qa_model.to(device)
ko_en_model = ko_en_model.to(device)
en_ko_model = en_ko_model.to(device)

def get_answer(
    question: str,
    model,
    tokenizer,
    ko_en_model,
    ko_en_tokenizer,
    en_ko_model,
    en_ko_tokenizer,
    context: str = None
) -> str:
    """질문에 대한 답변을 생성합니다."""
    try:
        # 언어 검증 로깅
        print("\n[언어 검증]")
        print("Q:", question)
        print("is_korean:", is_korean(question))
        print("is_english:", is_english(question))
        
        # 컨텍스트가 제공되지 않은 경우 관련 컨텍스트 검색
        if context is None:
            context = find_relevant_context(question)
            if not context:
                context = get_default_context(question)
        
        # 컨텍스트를 문장 단위로 분할
        sentences = sent_tokenize(context)
        
        # 각 문장의 중요도 평가
        sentence_scores = []
        for sentence in sentences:
            # 문장과 질문의 길이를 고려한 점수 계산
            score = len(sentence.split()) / (len(question.split()) + 1)
            sentence_scores.append((sentence, score))
        
        # 점수가 높은 순서로 정렬
        sentence_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 문장들을 선택하여 컨텍스트 구성
        selected_sentences = []
        current_length = 0
        MAX_TOKENS = 384  # FLAN-T5용
        
        for sentence, _ in sentence_scores:
            # 현재 문장을 추가했을 때의 토큰 수 예측
            test_context = ' '.join(selected_sentences + [sentence])
            test_inputs = tokenizer(test_context, return_tensors="pt", truncation=True, max_length=MAX_TOKENS)
            
            if len(test_inputs['input_ids'][0]) <= MAX_TOKENS:
                selected_sentences.append(sentence)
            else:
                break
        
        # 선택된 문장들로 컨텍스트 재구성
        context = ' '.join(selected_sentences)
        
        # 질문이 영어인 경우 한국어로 번역
        if not is_korean(question):
            print("\n[번역] 영어 -> 한국어")
            translated_question = translate_text(question, ko_en_model, ko_en_tokenizer)
            print("번역된 질문:", translated_question)
            question = translated_question
        
        # 프롬프트 생성
        prompt = f"""You are an AI assistant. Based on the context below, answer the question as clearly and concisely as possible.

Context:
{context}

Question:
{question}

Answer:"""
        
        # 답변 생성
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 원래 질문이 영어였다면 답변을 영어로 번역
        if not is_korean(question):
            print("\n[번역] 한국어 -> 영어")
            print("번역 전 답변:", answer)
            answer = translate_text(answer, en_ko_model, en_ko_tokenizer)
            print("번역 후 답변:", answer)
        
        return answer
        
    except Exception as e:
        print(f"답변 생성 중 오류 발생: {str(e)}")
        return "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."

def load_portfolio_data():
    """포트폴리오 데이터를 로드합니다."""
    texts = []
    
    # PDF 파일 처리 (하위 폴더 포함)
    pdf_files = glob.glob(os.path.join(PORTFOLIO_DATA_DIR, "**", "*.pdf"), recursive=True)
    print(f"\n발견된 PDF 파일: {pdf_files}")
    for pdf_file in pdf_files:
        texts.append(extract_text_from_pdf(pdf_file))
    
    # txt 파일 처리 (하위 폴더 포함)
    txt_files = glob.glob(os.path.join(PORTFOLIO_DATA_DIR, "**", "*.txt"), recursive=True)
    print(f"\n발견된 TXT 파일: {txt_files}")
    for txt_file in txt_files:
        print(f"\nTXT 파일 처리 중: {txt_file}")
        with open(txt_file, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    
    if not texts:
        print(f"경고: {PORTFOLIO_DATA_DIR}에서 PDF나 txt 파일을 찾을 수 없습니다.")
        print(f"현재 디렉토리: {os.getcwd()}")
        print(f"디렉토리 내용: {os.listdir(PORTFOLIO_DATA_DIR)}")
        for root, dirs, files in os.walk(PORTFOLIO_DATA_DIR):
            print(f"\n{root} 폴더 내용:")
            for file in files:
                print(f"- {file}")
    
    return texts

def extract_text_from_pdf(pdf_path):
    """PDF 파일에서 텍스트를 추출합니다."""
    print(f"\nPDF 파일 처리 중: {pdf_path}")
    text = ""
    try:
        # PyMuPDF를 사용하여 텍스트 추출
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        for i, page in enumerate(doc, 1):
            print(f"페이지 {i}/{total_pages} 처리 중...")
            # 텍스트 추출 시 레이아웃 정보 유지
            blocks = page.get_text("blocks")
            for block in blocks:
                if block[6] == 0:  # 텍스트 블록만 처리
                    text += block[4] + "\n"
        doc.close()
    except Exception as e:
        print(f"PyMuPDF 처리 중 오류 발생: {str(e)}")
        print("PyPDF2로 대체하여 처리를 시도합니다...")
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                for i, page in enumerate(reader.pages, 1):
                    print(f"페이지 {i}/{total_pages} 처리 중...")
                    text += page.extract_text() + "\n"
        except Exception as e2:
            print(f"PyPDF2 처리 중 오류 발생: {str(e2)}")
            return ""
    
    # 텍스트 정리
    text = clean_text(text)
    
    if not text:
        print(f"경고: {pdf_path}에서 텍스트를 추출할 수 없습니다.")
    else:
        print(f"추출된 텍스트 길이: {len(text)} 문자")
    
    return text

def clean_text(text):
    """텍스트를 정리합니다."""
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 특수 문자 처리 (영어 문자, 한글, 기본 문장 부호 보존)
    text = re.sub(r'[^\w\s가-힣a-zA-Z.,!?():\-]', '', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text

def create_vector_database(texts, embedding_model=None):
    """벡터 데이터베이스를 생성합니다."""
    try:
        print("\n벡터 DB 생성 중...")
        
        # 임베딩 모델 로드
        if embedding_model is None:
            print("임베딩 모델 로드 중...")
            embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        
        # 텍스트를 청크로 분할
        print("텍스트 청크 생성 중...")
        chunks = []
        for i, text in enumerate(texts):
            print(f"텍스트 {i+1}/{len(texts)} 처리 중...")
            text_chunks = create_chunks(text)
            print(f"텍스트 {i+1}에서 {len(text_chunks)}개의 청크 생성됨")
            chunks.extend(text_chunks)
        
        print(f"생성된 총 청크 수: {len(chunks)}")
        if not chunks:
            print("경고: 생성된 청크가 없습니다!")
            return False
            
        # 청크 품질 검사
        print("\n청크 품질 검사 중...")
        filtered_chunks = []
        for i, chunk in enumerate(chunks):
            # 최소 길이 확인
            if len(chunk) < 100:
                continue
            # 대문자로만 된 텍스트 제외
            if chunk.isupper():
                continue
            # 특수 문자가 과도하게 많은 텍스트 제외
            if len(re.findall(r'[^\w\s가-힣]', chunk)) / len(chunk) > 0.3:
                continue
            filtered_chunks.append(chunk)
            if (i + 1) % 100 == 0:
                print(f"청크 {i+1}/{len(chunks)} 검사 완료")
        
        chunks = filtered_chunks
        print(f"필터링 후 청크 수: {len(chunks)}")
        
        # 임베딩 생성 (배치 처리)
        print("\n임베딩 생성 중...")
        batch_size = 32
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            print(f"배치 {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} 처리 중...")
            batch_embeddings = embedding_model.encode(batch, show_progress_bar=True)
            embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        print(f"생성된 임베딩 shape: {embeddings.shape}")
        
        # FAISS 인덱스 생성
        print("\nFAISS 인덱스 생성 중...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Langchain FAISS 벡터 스토어 생성
        print("\nLangchain FAISS 벡터 스토어 생성 중...")
        embeddings_hf = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
        texts_metadata = [{"source": f"chunk_{i}"} for i in range(len(chunks))]
        db = FAISS.from_texts(chunks, embeddings_hf, metadatas=texts_metadata)
        
        # 저장
        print("\n벡터 DB 저장 중...")
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        faiss.write_index(index, os.path.join(VECTOR_DB_PATH, "faiss.index"))
        with open(os.path.join(VECTOR_DB_PATH, "chunks.pkl"), 'wb') as f:
            pickle.dump(chunks, f)
        
        # Langchain FAISS 벡터 스토어 저장
        db.save_local(os.path.join(VECTOR_DB_PATH, "langchain_faiss"))
        
        print("벡터 DB 생성 완료!")
        return True
        
    except Exception as e:
        print(f"벡터 DB 생성 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

def find_relevant_context(question: str, top_k: int = 10, threshold: float = 0.5) -> str:
    """질문과 관련된 컨텍스트를 찾습니다."""
    try:
        # 벡터 DB 파일 경로 확인
        vector_db_path = os.path.join(VECTOR_DB_PATH, "faiss.index")
        chunks_path = os.path.join(VECTOR_DB_PATH, "chunks.pkl")
        langchain_db_path = os.path.join(VECTOR_DB_PATH, "langchain_faiss")
        
        if not os.path.exists(vector_db_path) or not os.path.exists(chunks_path):
            print("벡터 DB 파일이 없습니다.")
            return ""
            
        # 임베딩 모델 로드
        embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        embeddings_hf = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
        
        # 인덱스와 청크 로드
        index = faiss.read_index(vector_db_path)
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        
        # Langchain FAISS 벡터 스토어 로드
        if os.path.exists(langchain_db_path):
            db = FAISS.load_local(
                folder_path=langchain_db_path,
                embeddings=embeddings_hf,
                allow_dangerous_deserialization=True
            )
        else:
            print("Langchain FAISS 벡터 스토어가 없습니다. 기본 검색을 사용합니다.")
            db = None
        
        # 질문 임베딩
        question_embedding = embedding_model.encode([question], normalize_embeddings=True)
        
        # 유사도 검색
        distances, indices = index.search(question_embedding.astype('float32'), top_k)
        
        # 디버깅 로그 추가
        print(f"[DEBUG] Top-k 검색 인덱스: {indices}")
        
        # 가장 관련성 높은 청크들 결합
        relevant_chunks = []
        similarity_scores = []
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= len(chunks):  # 인덱스 범위 체크
                continue
                
            similarity_score = 1 - dist/2  # L2 거리를 유사도 점수로 변환
            if similarity_score < threshold:
                continue
                
            relevant_chunks.append(chunks[idx])
            similarity_scores.append(similarity_score)
        
        # 디버깅 로그 추가
        print(f"[DEBUG] 초기 관련 청크:\n{relevant_chunks}")
        
        if not relevant_chunks:
            print("관련성 높은 컨텍스트를 찾을 수 없습니다.")
            return ""
        
        # MMR 적용
        if len(relevant_chunks) > 1:
            print("\n[MMR 적용 중]")
            candidate_embeddings = embedding_model.encode(relevant_chunks)
            relevant_chunks = mmr_rerank(question_embedding[0], candidate_embeddings, relevant_chunks, k=min(top_k, len(relevant_chunks)))
            print(f"MMR 적용 후 청크 수: {len(relevant_chunks)}")
            print(f"[DEBUG] MMR 적용 후 청크:\n{relevant_chunks}")
        
        # Re-ranking 적용 (reranker 모델이 있는 경우에만)
        try:
            reranker_model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
            reranker_tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            reranker_model = reranker_model.to(device)
            
            relevant_chunks = rerank_contexts(
                question, 
                relevant_chunks, 
                reranker_model, 
                reranker_tokenizer,
                device
            )
            print(f"[DEBUG] Re-ranking 적용 후 청크:\n{relevant_chunks}")
        except Exception as e:
            print(f"Re-ranking 적용 중 오류 발생: {str(e)}")
        
        print(f"찾은 컨텍스트 수: {len(relevant_chunks)}")
        if similarity_scores:
            print(f"최고 유사도 점수: {max(similarity_scores):.4f}")
        
        # 관련 청크 결합
        combined_context = " ".join(relevant_chunks)
        print("\n[DEBUG] 최종 선택된 컨텍스트:\n", combined_context)
        return combined_context
        
    except Exception as e:
        print(f"컨텍스트 검색 중 오류 발생: {str(e)}")
        return ""

def mmr_rerank(query_embedding, candidate_embeddings, candidates, k=5, lambda_param=0.5):
    """MMR(Maximal Marginal Relevance)을 사용하여 후보를 재순위화합니다."""
    selected_indices = []
    remaining_indices = list(range(len(candidates)))
    
    while len(selected_indices) < k and remaining_indices:
        # 선택된 후보와 남은 후보 간의 유사도 계산
        if selected_indices:
            selected_embeddings = candidate_embeddings[selected_indices]
            redundancy = np.max([
                np.dot(candidate_embeddings[i], selected_embeddings.T).mean()
                for i in remaining_indices
            ])
        else:
            redundancy = 0
        
        # MMR 점수 계산
        mmr_scores = []
        for i in remaining_indices:
            relevance = np.dot(candidate_embeddings[i], query_embedding)
            mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
            mmr_scores.append((i, mmr_score))
        
        # 최고 MMR 점수를 가진 후보 선택
        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    
    # 선택된 후보 반환
    return [candidates[i] for i in selected_indices]

def get_default_context(question: str) -> str:
    """기본 컨텍스트를 제공합니다."""
    # 질문에 따라 기본 컨텍스트 반환
    if "프로젝트" in question:
        return "프로젝트 경험: OCR 기술을 활용한 화장품 성분 분석, 비교 및 AI 화장품 추천 서비스 개발. 주요 업무: 한국어 기반 OCR 모델 개발, 성분 기반 화장품 Score 계산 알고리즘 개발, Cosine 유사도를 활용한 화장품 추천."
    elif "학력" in question or "자격증" in question:
        return "학력: 가천대학교 컴퓨터공학과 졸업 (2021.02-2025.02), 학점 3.58/4.5. 자격증: ADsP (데이터 분석 준전문가) (2024.11 취득), 네트워크 관리사 2급 (2019.01 취득)."
    else:
        return "죄송합니다. 관련 정보를 찾을 수 없습니다."

def is_korean(text: str) -> bool:
    """텍스트가 한국어인지 확인합니다."""
    # 한글 유니코드 범위: AC00-D7A3 (가-힣)
    korean_ratio = len([c for c in text if '\uAC00' <= c <= '\uD7A3']) / len(text) if text else 0
    return korean_ratio > 0.3  # 30% 이상이 한글이면 한국어로 판단

def is_english(text: str) -> bool:
    """텍스트가 영어인지 확인합니다."""
    # 영어 문자와 공백만 포함된지 확인
    english_ratio = len([c for c in text if c.isalpha() or c.isspace()]) / len(text) if text else 0
    return english_ratio > 0.8  # 80% 이상이 영어 문자면 영어로 판단

def rerank_contexts(question: str, contexts: List[str], model, tokenizer, device, top_k: int = 5) -> List[str]:
    """Re-ranking 모델을 사용하여 컨텍스트를 재순위화합니다."""
    try:
        # 질문과 컨텍스트 쌍 생성
        pairs = [[question, context] for context in contexts]
        
        # 토큰화
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # 점수 계산
        with torch.no_grad():
            scores = model(**inputs).logits.squeeze(-1)
        
        # 점수에 따라 컨텍스트 정렬
        scored_contexts = list(zip(contexts, scores.cpu().numpy()))
        scored_contexts.sort(key=lambda x: x[1], reverse=True)
        
        # 상위 k개 컨텍스트 반환
        return [context for context, _ in scored_contexts[:top_k]]
        
    except Exception as e:
        print(f"Re-ranking 중 오류 발생: {str(e)}")
        return contexts  # 오류 발생 시 원래 컨텍스트 반환

def create_chunks(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """텍스트를 청크로 분할합니다."""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        
        # 청크 추출
        chunk = text[start:end]
        
        # 문장 단위로 자르기
        sentences = sent_tokenize(chunk)
        if sentences:
            # 마지막 문장이 중간에 잘린 경우, 이전 문장까지만 포함
            if end < text_length and not text[end:end+1].isspace():
                chunks.append(' '.join(sentences[:-1]))
                start = end - len(sentences[-1])
            else:
                chunks.append(' '.join(sentences))
                start = end - overlap
        
    return chunks

def main():
    """메인 함수"""
    try:
        # 임베딩 모델 로드
        embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')

        # 포트폴리오 데이터 로드 및 벡터 데이터베이스 생성
        print("\n포트폴리오 데이터 로드 중...")
        portfolio_texts = load_portfolio_data()
        if portfolio_texts:
            create_vector_database(portfolio_texts, embedding_model)
        else:
            print("포트폴리오 데이터를 찾을 수 없습니다. 기본 컨텍스트를 사용합니다.")

        print("\n=== 자기소개 챗봇 테스트 시작 ===")

        # 테스트할 예시 질문들
        test_questions = [
            "프로젝트에 대해서 설명해주세요.",
            "학력 및 자격증을 말해주세요."
        ]

        # 각 질문에 대해 답변 생성
        for i, question in enumerate(test_questions, 1):
            print(f"\n[질문 {i}/{len(test_questions)}]")
            print(f"Q: {question}")

            try:
                # 답변 생성
                answer = get_answer(
                    question,
                    qa_model,
                    qa_tokenizer,
                    ko_en_model,
                    ko_en_tokenizer,
                    en_ko_model,
                    en_ko_tokenizer
                )
                print(f"A: {answer}")
            except Exception as e:
                print(f"답변 생성 중 오류 발생: {str(e)}")
                print("기본 답변을 사용합니다.")
                answer = get_default_context(question)
                print(f"A: {answer}")

            print("-" * 50)

        print("\n=== 테스트 완료 ===")

        # 질문과 최종 답변 정리
        print("\n=== 질문과 최종 답변 정리 ===")
        for i, (question, answer) in enumerate(zip(
            test_questions,
            [
                get_answer(q, qa_model, qa_tokenizer, ko_en_model, ko_en_tokenizer, en_ko_model, en_ko_tokenizer)
                for q in test_questions
            ]
        ), 1):
            print(f"질문 {i}: {question}")
            print(f"답변 {i}: {answer}")
            print("-" * 50)

    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        
if __name__ == "__main__":
    main()
