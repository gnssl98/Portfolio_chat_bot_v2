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
from typing import List, Tuple, Dict, Optional
import nltk
from nltk.tokenize import sent_tokenize
from peft import PeftModel, PeftConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import pipeline
import spacy
from concurrent.futures import ThreadPoolExecutor
import gc
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 파일 핸들러 추가 (로그를 파일에도 저장)
file_handler = logging.FileHandler('training.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# spaCy 모델 로드
try:
    nlp = spacy.load('ko_core_news_sm', disable=['ner', 'tagger', 'parser'])
    nlp.add_pipe('sentencizer')
except OSError:
    spacy.cli.download('ko_core_news_sm')
    nlp = spacy.load('ko_core_news_sm', disable=['ner', 'tagger', 'parser'])
    nlp.add_pipe('sentencizer')

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

# 토크나이저 정의
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

# 임베딩 모델 이름 상수 추가
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"
EMBEDDING_DEVICE = "cuda"

def get_token_length(text: str, tokenizer=None) -> int:
    """텍스트의 토큰 길이를 반환합니다."""
    tokenizer = tokenizer or AutoTokenizer.from_pretrained("google/flan-t5-large")
    return len(tokenizer(text)["input_ids"])

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
        # 원본 질문 언어 확인 및 저장
        is_question_korean = is_korean(question)
        is_question_english = is_english(question)
        
        print("\n[언어 검증]")
        print("Q:", question)
        print("is_korean:", is_question_korean)
        print("is_english:", is_question_english)
        
        # 영어 질문을 한국어로 번역 (컨텍스트 검색용)
        working_question = question
        if is_question_english:
            print("\n[번역] 영어 -> 한국어 (컨텍스트 검색용)")
            working_question = translate_text(question, en_ko_model, en_ko_tokenizer)
            print("번역된 질문:", working_question)
        
        # 컨텍스트 검색 및 처리
        if context is None:
            try:
                print("\n[컨텍스트 검색]")
                context = find_relevant_context(working_question)
                if not context or not context.strip():
                    print("벡터 검색 실패 → 기본 컨텍스트 사용")
                    context = get_default_context(working_question)
            except Exception as e:
                print(f"컨텍스트 검색 중 오류 발생: {str(e)} → 기본 컨텍스트 사용")
                context = get_default_context(working_question)
        
        print("\n[컨텍스트 미리보기]")
        print(context[:300] + "..." if len(context) > 300 else context)
        
        # 컨텍스트를 영어로 번역
        print("\n[컨텍스트 번역] 한국어 -> 영어")
        english_context = translate_text(context, ko_en_model, ko_en_tokenizer)
        print("번역된 컨텍스트:", english_context[:300] + "..." if len(english_context) > 300 else english_context)
        
        # 컨텍스트 길이 제한
        MAX_CONTEXT_TOKENS = 350
        print("\n[컨텍스트 길이 제한]")
        print(f"원본 컨텍스트 토큰 수: {get_token_length(english_context)}")
        context_inputs = tokenizer(english_context, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT_TOKENS)
        english_context = tokenizer.decode(context_inputs['input_ids'][0], skip_special_tokens=True)
        print(f"제한된 컨텍스트 토큰 수: {get_token_length(english_context)}")
        
        # 질문을 영어로 번역 (QA 모델 입력용)
        if is_question_korean:
            print("\n[번역] 한국어 -> 영어 (QA 모델용)")
            english_question = translate_text(question, ko_en_model, ko_en_tokenizer)
            print("번역된 질문:", english_question)
        else:
            english_question = question
        
        # 영어 프롬프트 생성
        prompt = f"""You are a helpful assistant answering questions about a portfolio.

Based on the following context, answer the question clearly and specifically.

Context:
{english_context}

Question:
{english_question}

Answer:"""
        
        print("\n[PROMPT PREVIEW]")
        print(prompt[:300] + "..." if len(prompt) > 300 else prompt)
        print(f"\n프롬프트 토큰 수: {get_token_length(prompt)}")
        
        # 답변 생성 (영어로)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_beams=5,
            no_repeat_ngram_size=2,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            early_stopping=True
        )
        
        english_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("\n[생성된 영어 답변]")
        print(english_answer)
        
        # 답변을 원본 질문 언어로 번역
        if is_question_korean:
            print("\n[번역] 영어 -> 한국어")
            answer = translate_text(english_answer, en_ko_model, en_ko_tokenizer)
            print("번역된 답변:", answer)
        else:
            answer = english_answer
        
        return answer
        
    except Exception as e:
        print(f"답변 생성 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
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

def is_chunk_valid(chunk: str) -> bool:
    """청크의 유효성을 검사합니다."""
    # 기본 검사
    if not chunk or not chunk.strip():
        return False
    
    # 단어 수가 너무 적은 경우 제외
    if len(chunk.split()) < 10:
        return False
    
    # 모두 대문자인 경우 제외 (헤더나 메타데이터일 가능성)
    if chunk.isupper():
        return False
    
    # URL 포함 청크는 제외
    if 'http' in chunk or 'www.' in chunk:
        return False
    
    # 숫자 비율이 너무 높은 경우 제외 (ex. 성적표)
    digit_ratio = len([c for c in chunk if c.isdigit()]) / len(chunk)
    if digit_ratio > 0.5:  # 50% 이상이 숫자면 제외
        return False
    
    # 특수문자나 숫자가 과도하게 많은 경우 제외
    special_char_ratio = len([c for c in chunk if not c.isalnum() and not c.isspace()]) / len(chunk)
    if special_char_ratio > 0.3:  # 30% 이상이 특수문자면 제외
        return False
    
    # 한글이나 영어 문자가 너무 적은 경우 제외
    text_char_ratio = len([c for c in chunk if c.isalpha()]) / len(chunk)
    if text_char_ratio < 0.5:  # 50% 미만이 텍스트면 제외
            return False
            
    # 디버깅을 위한 비율 출력
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"\n[청크 검증 정보]")
        logger.debug(f"청크 길이: {len(chunk)}")
        logger.debug(f"단어 수: {len(chunk.split())}")
        logger.debug(f"숫자 비율: {digit_ratio:.2f}")
        logger.debug(f"특수문자 비율: {special_char_ratio:.2f}")
        logger.debug(f"텍스트 비율: {text_char_ratio:.2f}")
        if digit_ratio > 0.3:  # 숫자가 30% 이상인 경우 내용 출력
            logger.debug(f"높은 숫자 비율 청크: {chunk[:100]}...")
    
    return True

def create_chunks(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """텍스트를 토큰 기반 청크로 분할하는 최적화된 함수"""
    if not text or not text.strip():
        return []
    
    # 문장 분리
    doc = nlp(text)
    sentences = [str(sent).strip() for sent in doc.sents]
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        # 토큰 길이 계산
        sentence_tokens = get_token_length(sentence)
        
        if current_size + sentence_tokens > chunk_size and current_chunk:
            # 청크 생성
            chunk_text = ' '.join(current_chunk)
            # 유효성 검사 후 추가
            if is_chunk_valid(chunk_text):
                chunks.append(chunk_text)
            
            # 오버랩을 위해 마지막 부분 유지
            overlap_size = 0
            overlap_chunk = []
            for sent in reversed(current_chunk):
                sent_tokens = get_token_length(sent)
                if overlap_size + sent_tokens > overlap:
                    break
                overlap_chunk.insert(0, sent)
                overlap_size += sent_tokens
            
            current_chunk = overlap_chunk
            current_size = overlap_size
        
        current_chunk.append(sentence)
        current_size += sentence_tokens
    
    # 마지막 청크 처리
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if is_chunk_valid(chunk_text):
            chunks.append(chunk_text)
    
    # 청크 생성 정보 출력
    print(f"\n[청크 생성 정보]")
    print(f"총 청크 수: {len(chunks)}")
    for i, chunk in enumerate(chunks):
        chunk_tokens = get_token_length(chunk)
        print(f"청크 {i+1}: {chunk_tokens} 토큰, {len(chunk.split())} 단어")
    
    # 메모리 정리
    gc.collect()
    
    return chunks

def create_vector_database(texts: List[str], embedding_model: HuggingFaceEmbeddings) -> Optional[FAISS]:
    """벡터 데이터베이스를 생성합니다."""
    try:
        print("\n[벡터 데이터베이스 생성]")
        print(f"임베딩 모델: {embedding_model.model_name}")
        
        if not texts:
            print("경고: 입력 텍스트가 없습니다.")
            return None
            
        # 청크 생성 및 필터링
        print("텍스트 청크 생성 중...")
        chunks = []
        for i, text in enumerate(texts, 1):
            if not text or not isinstance(text, str):
                print(f"경고: 텍스트 {i}가 비어있거나 문자열이 아닙니다. 건너뜁니다.")
                continue
                
            try:
                text_chunks = create_chunks(text)
                valid_chunks = [chunk for chunk in text_chunks if is_chunk_valid(chunk)]
                chunks.extend(valid_chunks)
                print(f"텍스트 {i}: {len(valid_chunks)}개의 유효한 청크 생성")
            except Exception as e:
                print(f"텍스트 {i} 청크 생성 중 오류 발생: {str(e)}")
                continue
        
        if not chunks:
            print("경고: 유효한 청크가 없습니다.")
            return None
            
        print(f"총 청크 수: {len(chunks)}")
        
        # 벡터 스토어 생성
        try:
            print("FAISS 벡터 스토어 생성 중...")
            vectorstore = FAISS.from_texts(
                texts=chunks,
                embedding=embedding_model
            )
            
            # 벡터 스토어 저장
            print(f"벡터 스토어를 {VECTOR_DB_PATH}에 저장 중...")
            vectorstore.save_local(VECTOR_DB_PATH)
            print("벡터 데이터베이스 생성 완료")
            return vectorstore
            
        except Exception as e:
            print(f"FAISS 벡터 스토어 생성/저장 중 오류: {str(e)}")
            raise
            
    except Exception as e:
        print(f"벡터 데이터베이스 생성 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def find_relevant_context(question: str, top_k: int = 5) -> str:
    """질문과 관련된 컨텍스트를 찾습니다."""
    try:
        if not question.strip():
            print("질문이 비어있습니다.")
            return ""
            
        if not os.path.exists(VECTOR_DB_PATH):
            print(f"벡터 데이터베이스를 찾을 수 없습니다: {VECTOR_DB_PATH}")
            return ""
            
        # 임베딩 모델 로드
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        try:
            # allow_dangerous_deserialization 파라미터 추가
            vectorstore = FAISS.load_local(
                VECTOR_DB_PATH, 
                embedding_model,
                allow_dangerous_deserialization=True  # 추가
            )
        except Exception as e:
            print(f"벡터 스토어 로드 실패: {str(e)}")
            return ""
        
        # 유사도 검색
        try:
            print("유사도 검색 수행 중...")
            docs = vectorstore.similarity_search_with_score(question, k=top_k)
        except Exception as e:
            print(f"유사도 검색 실패: {str(e)}")
            return ""
            
        if not docs:
            print("관련 문서를 찾을 수 없습니다.")
            return ""
            
        # 문서 내용과 유사도 점수 결합
        print("\n[검색 결과]")
        contexts = []
        for i, (doc, score) in enumerate(docs, 1):
            print(f"문서 {i} (유사도 점수: {score:.4f})")
            if score < 1.5:  # 유사도 점수가 좋은 문서만 선택
                contexts.append(doc.page_content)
                print("- 선택됨")
            else:
                print("- 유사도 점수가 너무 낮아 제외됨")
        
        if not contexts:
            print("유사도 기준을 만족하는 문서가 없습니다.")
            return ""
            
        print(f"\n선택된 문서 수: {len(contexts)}")
        return " ".join(contexts)
        
    except Exception as e:
        print(f"컨텍스트 검색 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
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
    """텍스트가 한국어인지 판단합니다.
    한글 비율이 30% 이상이면 한국어로 간주합니다."""
    if not text or not isinstance(text, str):
        return False
    korean_ratio = sum('\uAC00' <= c <= '\uD7A3' for c in text) / len(text)
    return korean_ratio > 0.3

def is_english(text: str) -> bool:
    """텍스트가 영어인지 판단합니다.
    한글이 아니면서 알파벳과 공백 비율이 80% 이상이면 영어로 간주합니다."""
    if not text or not isinstance(text, str):
        return False
    if is_korean(text):  # 한글로 판단되면 영어로 간주하지 않음
        return False
    english_ratio = sum(c.isalpha() or c.isspace() for c in text) / len(text)
    return english_ratio > 0.8

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

def translate_text(text: str, model, tokenizer) -> str:
    """텍스트를 번역합니다."""
    try:
        # 입력 텍스트가 비어있는 경우
        if not text or not text.strip():
            return text
        
        # 한->영 또는 영->한 설정
        if is_korean(text):
            tokenizer.src_lang = "ko_KR"
            tokenizer.tgt_lang = "en_XX"
            forced_bos_token_id = tokenizer.lang_code_to_id["en_XX"]
        else:
            tokenizer.src_lang = "en_XX"
            tokenizer.tgt_lang = "ko_KR"
            forced_bos_token_id = tokenizer.lang_code_to_id["ko_KR"]
        
        # 디버깅 정보 출력
        print(f"\n[번역 설정]")
        print(f"원본 텍스트: {text[:100]}...")
        print(f"원본 언어: {tokenizer.src_lang}")
        print(f"목표 언어: {tokenizer.tgt_lang}")
        print(f"forced_bos_token_id: {forced_bos_token_id}")
        
        # 텍스트 토큰화
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        ).to(device)
        
        # 번역 생성
        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,
                length_penalty=1.0,
                temperature=0.3,
                do_sample=False,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
        
        # 번역 결과 디코딩
        result = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
        
        print(f"번역 결과: {result[:100]}...")
        
        return result
        
    except Exception as e:
        print(f"번역 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return text  # 오류 발생 시 원본 텍스트 반환

def main():
    """메인 함수"""
    try:
        # 임베딩 모델 초기화
        print("\n[초기화]")
        print("임베딩 모델 초기화 중...")
        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True}
        )
        print("임베딩 모델 초기화 완료")
        
        # 포트폴리오 데이터 로드 및 벡터 데이터베이스 생성
        print("\n[데이터 로드]")
        portfolio_texts = load_portfolio_data()
        if portfolio_texts:
            print(f"포트폴리오 텍스트 {len(portfolio_texts)}개 로드됨")
            create_vector_database(portfolio_texts, embedding_model)
        else:
            print("포트폴리오 데이터를 찾을 수 없습니다. 기본 컨텍스트를 사용합니다.")

        print("\n=== 자기소개 챗봇 테스트 시작 ===")

        # 테스트할 예시 질문들
        test_questions = [
            "프로젝트에 대해서 설명해주세요.",
            "학력 및 자격증을 말해주세요."
        ]

        # 질문과 답변 결과를 저장할 리스트
        results = []

        # 각 질문에 대해 답변 생성 및 저장
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
                results.append((question, answer))
            except Exception as e:
                print(f"답변 생성 중 오류 발생: {str(e)}")
                print("기본 답변을 사용합니다.")
                default_answer = get_default_context(question)
                print(f"A: {default_answer}")
                results.append((question, default_answer))

            print("-" * 50)

        print("\n=== 테스트 완료 ===")

        # 저장된 결과를 사용하여 최종 정리
        print("\n=== 질문과 최종 답변 정리 ===")
        for i, (question, answer) in enumerate(results, 1):
            print(f"\n질문 {i}: {question}")
            print(f"답변 {i}: {answer}")
            print("-" * 50)

    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
if __name__ == "__main__":
    main()
