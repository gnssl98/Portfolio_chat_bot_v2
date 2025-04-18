import os
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerFast,
    MBartForConditionalGeneration,
    MBart50TokenizerFast
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
from typing import List, Tuple
import nltk
from nltk.tokenize import sent_tokenize

# NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# 환경 변수 로드
load_dotenv()

# 경로 설정
MODEL_ID = "google/flan-t5-large"  # Hugging Face 모델 ID
FINETUNED_MODEL_PATH = r"D:\dataset\fine_tuned_model\flan-t5-large-finetuned"  # 파인튜닝된 모델 경로
CACHE_DIR = r"D:\dataset\huggingface_cache"
VECTOR_DB_PATH = "./vector_db"
TRANSLATION_MODEL_PATH = "./translation_model"
PORTFOLIO_DATA_DIR = "./portfolio_data"

def clean_text(text):
    """텍스트를 정리합니다."""
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 특수 문자 처리 (영어 문자, 한글, 기본 문장 부호 보존)
    text = re.sub(r'[^\w\s가-힣a-zA-Z.,!?():\-]', '', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text

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

def create_chunks(text: str, min_chunk_size: int = 100, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """텍스트를 청크로 분할합니다."""
    # 문장 단위로 분리
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:  # 빈 문장 건너뛰기
            continue
            
        # 문장이 너무 긴 경우 분할
        if len(sentence) > max_chunk_size:
            words = sentence.split()
            temp_chunk = ""
            for word in words:
                if len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                    temp_chunk += word + " "
                else:
                    if len(temp_chunk) >= min_chunk_size:
                        chunks.append(temp_chunk.strip())
                    temp_chunk = word + " "
            if temp_chunk and len(temp_chunk) >= min_chunk_size:
                chunks.append(temp_chunk.strip())
            continue
            
        # 현재 청크에 문장 추가 시도
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            # 현재 청크가 최소 크기를 넘으면 저장
            if len(current_chunk) >= min_chunk_size:
                chunks.append(current_chunk.strip())
            # 오버랩을 위해 마지막 문장들 유지
            words = current_chunk.split()
            overlap_text = " ".join(words[-int(len(words)*0.2):])  # 20% 오버랩
            current_chunk = overlap_text + " " + sentence + " "
    
    # 마지막 청크 처리
    if current_chunk and len(current_chunk) >= min_chunk_size:
        chunks.append(current_chunk.strip())
    
    return chunks

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
        for text in texts:
            chunks.extend(create_chunks(text))
        
        print(f"생성된 청크 수: {len(chunks)}")
        if not chunks:
            print("경고: 생성된 청크가 없습니다!")
            return False
            
        # 청크 품질 검사
        filtered_chunks = []
        for chunk in chunks:
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
        
        chunks = filtered_chunks
        print(f"필터링 후 청크 수: {len(chunks)}")
        
        # 임베딩 생성
        print("임베딩 생성 중...")
        embeddings = embedding_model.encode(chunks, show_progress_bar=True)
        
        # FAISS 인덱스 생성
        print("FAISS 인덱스 생성 중...")
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # 저장
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        faiss.write_index(index, os.path.join(VECTOR_DB_PATH, "faiss.index"))
        with open(os.path.join(VECTOR_DB_PATH, "chunks.pkl"), 'wb') as f:
            pickle.dump(chunks, f)
            
        return True
        
    except Exception as e:
        print(f"벡터 DB 생성 중 오류 발생: {str(e)}")
        return False

def find_relevant_context(question: str, top_k: int = 3, threshold: float = 0.7) -> Tuple[str, float]:
    """질문과 관련된 컨텍스트를 찾습니다."""
    try:
        # 벡터 DB 파일 경로 확인
        vector_db_path = os.path.join(VECTOR_DB_PATH, "faiss.index")
        chunks_path = os.path.join(VECTOR_DB_PATH, "chunks.pkl")
        
        if not os.path.exists(vector_db_path) or not os.path.exists(chunks_path):
            print("벡터 DB 파일이 없습니다.")
            return "", 0.0
            
        # 임베딩 모델 로드
        embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        
        # 인덱스와 청크 로드
        index = faiss.read_index(vector_db_path)
        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)
        
        # 질문 임베딩
        question_embedding = embedding_model.encode([question])
        
        # 유사도 검색
        distances, indices = index.search(question_embedding.astype('float32'), top_k)
        
        # 가장 관련성 높은 청크들 결합
        relevant_chunks = []
        max_similarity = 0.0
        
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= len(chunks):  # 인덱스 범위 체크
                continue
                
            similarity_score = 1 - dist/2  # L2 거리를 유사도 점수로 변환
            if similarity_score < threshold:
                continue
                
            if i == 0:  # 첫 번째 청크의 유사도 저장
                max_similarity = similarity_score
                
            relevant_chunks.append(chunks[idx])
        
        if not relevant_chunks:
            print("관련성 높은 컨텍스트를 찾을 수 없습니다.")
            return "", 0.0
            
        # 관련 청크들을 하나의 컨텍스트로 결합
        context = " ".join(relevant_chunks)
        print(f"찾은 컨텍스트 수: {len(relevant_chunks)}")
        print(f"최고 유사도 점수: {max_similarity:.4f}")
        
        return context, max_similarity
        
    except Exception as e:
        print(f"컨텍스트 검색 중 오류 발생: {str(e)}")
        return "", 0.0

def get_answer(question, model, tokenizer, device, use_context=False):
    """질문에 대한 답변을 생성합니다."""
    try:
        # 입력 텍스트 전처리
        question = question.strip()
        if not question:
            return "질문을 입력해주세요."
        
        print("\n[1. 질문 전처리]")
        print(f"원본 질문: {question}")
        
        # 1. 한국어 질문을 영어로 번역
        print("\n[2. 질문 번역]")
        english_question = translate_text(question, "ko", "en")
        print(f"번역된 질문: {english_question}")
        
        # 2. 컨텍스트 검색 및 번역
        context = None
        english_context = ""
        if use_context:
            context, similarity = find_relevant_context(question)  # 튜플 언패킹
            if context and similarity >= 0.7:  # 유사도가 0.7 이상인 경우만 사용
                # 컨텍스트 길이 제한 (500자)
                if len(context) > 500:
                    context = context[:500] + "..."
                english_context = translate_text(context, "ko", "en")
                print(f"\n컨텍스트 (유사도: {similarity:.4f}):\n{english_context}")
        
        # 3. 프롬프트 구성
        print("\n[3. 프롬프트 구성]")
        if context and similarity >= 0.7:
            input_text = f"""Context: {english_context}

Question: {english_question}

Instructions:
1. Answer the question directly and concisely based on the context.
2. If you can't find the exact information, use the most relevant information from the context.
3. Keep your answer short and to the point.
4. If you really don't have enough information, say "I don't have enough information to answer that question."

Answer:"""
        else:
            input_text = f"""Question: {english_question}

Instructions:
1. Answer the question directly and concisely.
2. Keep your answer short and to the point.
3. If you don't have enough information, use the default response.

Answer:"""
        
        print("\n[모델 입력 텍스트]")
        print(input_text)
        
        # 4. 토큰화 및 생성
        print("\n[4. 답변 생성]")
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=150,  # 답변 길이 제한
                min_length=10,   # 최소 길이 설정
                num_beams=5,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                early_stopping=True
            )
        
        english_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 답변에서 "Answer:" 이후 부분만 추출 (개선된 방식)
        if "Answer:" in english_answer:
            english_answer = english_answer.split("Answer:")[-1].strip()
        else:
            english_answer = english_answer.strip()
        
        # 답변이 너무 짧으면 기본 답변 사용
        if len(english_answer) < 5:
            return get_default_context(question)
            
        print(f"\n영어 답변: {english_answer}")
        
        # 5. 영어 답변을 한국어로 번역
        print("\n[5. 답변 번역]")
        
        # 기본 답변 처리
        if "don't have enough information" in english_answer.lower():
            korean_answer = "죄송합니다. 해당 질문에 대한 정보가 충분하지 않습니다."
        else:
            korean_answer = translate_text(english_answer, "en", "ko")
            
            # 번역 결과가 비어있거나 너무 짧은 경우 기본 답변 사용
            if not korean_answer or len(korean_answer) < 5:
                print("번역 결과가 비어있거나 너무 짧습니다. 기본 답변을 사용합니다.")
                korean_answer = get_default_context(question)
        
        print(f"한국어 답변: {korean_answer}")
        
        return korean_answer
        
    except Exception as e:
        print(f"답변 생성 중 오류 발생: {str(e)}")
        return get_default_context(question)  # 오류 발생 시 기본 답변 사용

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

def load_models():
    """모든 필요한 모델을 로드합니다."""
    print("모델 로드 중...")
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")
    
    # QA 모델 로드 (파인튜닝된 모델 사용)
    print(f"파인튜닝된 QA 모델 로드 중: {FINETUNED_MODEL_PATH}")
    try:
        qa_model = AutoModelForSeq2SeqLM.from_pretrained(
            FINETUNED_MODEL_PATH,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR
        ).to(device)
        
        qa_tokenizer = AutoTokenizer.from_pretrained(
            FINETUNED_MODEL_PATH,
            cache_dir=CACHE_DIR
        )
        print("✓ 파인튜닝된 모델 로드 완료")
    except Exception as e:
        print(f"파인튜닝된 모델 로드 실패: {str(e)}")
        print("기본 모델로 대체합니다.")
        qa_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR
        ).to(device)
        
        qa_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR
        )
    
    # 번역 모델 로드 (원본 MBART 모델 사용)
    translation_model = MBartForConditionalGeneration.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    ).to(device)
    
    translation_tokenizer = MBart50TokenizerFast.from_pretrained(
        "facebook/mbart-large-50-many-to-many-mmt"
    )
    
    # 번역 모델 토크나이저 디버깅
    print("\n[번역 모델 토크나이저 디버깅]")
    print("사용 가능한 언어 코드:")
    print(translation_tokenizer.lang_code_to_id)
    
    # 임베딩 모델 로드 (e5-large)
    embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
    
    print("✓ 모든 모델 로드 완료")
    
    return {
        'qa_model': qa_model,
        'qa_tokenizer': qa_tokenizer,
        'translation_model': translation_model,
        'translation_tokenizer': translation_tokenizer,
        'embedding_model': embedding_model,
        'device': device
    }

def translate_text(text, src_lang="ko", tgt_lang="en"):
    """텍스트를 번역합니다."""
    try:
        global models  # 전역 변수 참조
        
        if not text:
            print("번역할 텍스트가 비어있습니다.")
            return text
            
        # 입력 텍스트 정리
        text = clean_text(text)
        
        # 소스 언어와 목표 언어가 같으면 번역하지 않음
        if src_lang == tgt_lang:
            return text
            
        # 소스 언어 설정 (ko -> ko_KR, en -> en_XX)
        src_lang_code = "ko_KR" if src_lang == "ko" else "en_XX"
        tgt_lang_code = "en_XX" if tgt_lang == "en" else "ko_KR"
        
        # 디버깅용 로그 추가
        print("\n[번역 설정]")
        print(f"원본 텍스트: {text[:100]}...")  # 앞부분 100자만 출력
        print(f"소스 언어: {src_lang_code}, 목표 언어: {tgt_lang_code}")
        
        if 'translation_tokenizer' not in models or 'translation_model' not in models:
            print("번역 모델이 로드되지 않았습니다.")
            return text
        
        # 토크나이저 설정
        try:
            models['translation_tokenizer'].src_lang = src_lang_code
        except Exception as e:
            print(f"토크나이저 언어 설정 중 오류 발생: {str(e)}")
            return text
        
        # 목표 언어의 토큰 ID 가져오기
        try:
            lang_id = models['translation_tokenizer'].lang_code_to_id[tgt_lang_code]
        except (KeyError, AttributeError) as e:
            print(f"언어 코드 {tgt_lang_code}에 대한 토큰 ID를 찾을 수 없습니다: {str(e)}")
            return text
            
        # 토크나이징
        try:
            encoded = models['translation_tokenizer'](
                text, 
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(models['device'])
        except Exception as e:
            print(f"토크나이징 중 오류 발생: {str(e)}")
            return text
        
        # 번역 생성
        try:
            with torch.no_grad():
                generated_tokens = models['translation_model'].generate(
                    **encoded,
                    max_length=512,
                    num_beams=5,
                    length_penalty=1.0,
                    temperature=0.8,
                    do_sample=True,
                    top_p=0.95,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                    forced_bos_token_id=lang_id,  # 목표 언어 토큰 ID 설정
                    bad_words_ids=[[models['translation_tokenizer'].unk_token_id]]  # UNK 토큰 방지
                )
        except Exception as e:
            print(f"번역 생성 중 오류 발생: {str(e)}")
            return text
        
        # 번역 결과 디코딩 및 정리
        try:
            translated = models['translation_tokenizer'].batch_decode(
                generated_tokens, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0]
            
            # 번역 결과 정리
            translated = clean_text(translated)
            
            # 번역 결과 검증
            if not translated or translated.strip() == "":
                print("번역 결과가 비어있습니다.")
                return text
                
            if translated == text:
                print("번역 결과가 원본과 동일합니다.")
                return text
                
            # 번역 결과에 원본 언어가 섞여 있는지 확인
            if tgt_lang == "en" and any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in translated):
                print("영어 번역 결과에 한글이 포함되어 있습니다.")
                return text
                
            if tgt_lang == "ko" and all(ord(char) < 0xAC00 or ord(char) > 0xD7A3 for char in translated):
                print("한글 번역 결과에 한글이 없습니다.")
                return text
                
            print(f"번역 결과: {translated[:100]}...")  # 앞부분 100자만 출력
            return translated
            
        except Exception as e:
            print(f"번역 결과 디코딩 중 오류 발생: {str(e)}")
            return text
        
    except Exception as e:
        print(f"번역 중 오류 발생: {str(e)}")
        return text

def get_default_context(question):
    """기본 컨텍스트를 반환합니다."""
    # 질문 유형에 따른 기본 컨텍스트
    if "이름" in question or "누구" in question:
        return "저는 안창훈입니다."
    elif "직업" in question or "일" in question:
        return "저는 AI 엔지니어입니다."
    elif "학력" in question or "학교" in question:
        return "저는 가천대학교 컴퓨터공학과를 졸업했습니다."
    elif "경력" in question or "경험" in question:
        return "저는 AI 연구 및 개발 경험이 있습니다."
    elif "기술" in question or "스택" in question:
        return "저는 Python, JavaScript, 머신러닝, 딥러닝 기술을 보유하고 있습니다."
    elif "프로젝트" in question:
        return "저는 AI 기반 프로젝트와 웹 애플리케이션 개발 경험이 있습니다."
    elif "취미" in question:
        return "저의 취미는 코딩과 새로운 기술을 배우는 것입니다."
    elif "목표" in question:
        return "저의 목표는 AI 분야에서 전문가가 되어 혁신적인 솔루션을 개발하는 것입니다."
    elif "강점" in question:
        return "저의 강점은 문제 해결 능력과 새로운 기술을 빠르게 배우는 능력입니다."
    elif "약점" in question:
        return "저는 때때로 완벽주의적 경향이 있어 작업을 완료하는 데 시간이 더 걸릴 수 있습니다."
    else:
        return "저는 AI 엔지니어로서 다양한 기술 스택과 프로젝트 경험을 가지고 있습니다."

def main():
    """메인 함수"""
    try:
        global models
        models = load_models()
        
        # 포트폴리오 데이터 로드 및 벡터 데이터베이스 생성
        print("\n포트폴리오 데이터 로드 중...")
        portfolio_texts = load_portfolio_data()
        if portfolio_texts:
            create_vector_database(portfolio_texts, models['embedding_model'])
        else:
            print("포트폴리오 데이터를 찾을 수 없습니다. 기본 컨텍스트를 사용합니다.")
        
        print("\n=== 자기소개 챗봇 테스트 시작 ===")
        
        # 테스트할 예시 질문들
        test_questions = [
            "당신의 이름은 무엇인가요?",
            "당신의 직업은 무엇인가요?",
            "당신의 학력은 어떻게 되나요?",
            "당신의 경력은 어떻게 되나요?",
            "당신의 기술 스택은 무엇인가요?",
            "당신의 프로젝트 경험을 설명해주세요.",
            "당신의 취미는 무엇인가요?",
            "당신의 목표는 무엇인가요?",
            "당신의 강점은 무엇인가요?",
            "당신의 약점은 무엇인가요?"
        ]
        
        # 각 질문에 대해 답변 생성
        for i, question in enumerate(test_questions, 1):
            print(f"\n[질문 {i}/10]")
            print(f"Q: {question}")
            
            try:
                # 답변 생성
                answer = get_answer(
                    question=question,
                    model=models['qa_model'],
                    tokenizer=models['qa_tokenizer'],
                    device=models['device'],
                    use_context=True
                )
                print(f"A: {answer}")
            except Exception as e:
                print(f"답변 생성 중 오류 발생: {str(e)}")
                print("기본 답변을 사용합니다.")
                answer = get_default_context(question)
                print(f"A: {answer}")
            
            print("-" * 50)
        
        print("\n=== 테스트 완료 ===")
            
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 