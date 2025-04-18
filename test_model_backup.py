import os
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizerFast
)
import torch
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
import re
import glob
import PyPDF2
import json

# 환경 변수 로드
load_dotenv()

# 경로 설정
MODEL_ID = "google/flan-t5-xl"  # Hugging Face 모델 ID
LORA_WEIGHTS_PATH = r"D:\dataset\fine_tuned_model\flan-t5-xl-finetuned\lora_weights"
CACHE_DIR = r"D:\dataset\huggingface_cache"
VECTOR_DB_PATH = "./vector_db"
TRANSLATION_MODEL_PATH = "./translation_model"
PORTFOLIO_DATA_DIR = "./portfolio_data"

def clean_text(text):
    """텍스트를 정리합니다."""
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text)
    # 특수 문자 처리 (영어 문자 보존)
    text = re.sub(r'[^\w\s가-힣a-zA-Z.,!?]', '', text)
    # 앞뒤 공백 제거
    text = text.strip()
    return text

def extract_text_from_pdf(pdf_path):
    """PDF 파일에서 텍스트를 추출합니다."""
    print(f"\nPDF 파일 처리 중: {pdf_path}")
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdf_reader.pages, 1):
                print(f"페이지 {i}/{len(pdf_reader.pages)} 처리 중...")
                page_text = page.extract_text()
                if page_text.strip():  # 빈 페이지 제외
                    text += page_text + "\n\n"
    except Exception as e:
        print(f"PDF 처리 중 오류 발생: {str(e)}")
    
    # 텍스트 정리
    text = re.sub(r'\s+', ' ', text)  # 연속된 공백 제거
    text = text.strip()
    
    if not text:
        print(f"경고: {pdf_path}에서 텍스트를 추출할 수 없습니다.")
    else:
        print(f"추출된 텍스트 길이: {len(text)} 문자")
    
    return text

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

def create_vector_database(texts, embedding_model=None):
    """벡터 데이터베이스 생성"""
    try:
        # 임베딩 모델 로드
        if embedding_model is None:
            embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
        
        # 텍스트 청크 생성 (overlap 적용)
        chunks = []
        min_chunk_size = 20  # 최소 청크 크기 축소
        max_chunk_size = 400  # 최대 청크 크기 축소
        overlap = 50  # 오버랩 크기 축소
        
        for text in texts:
            # 문단 단위로 분할
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            for paragraph in paragraphs:
                # 문장 단위로 분할
                sentences = [s.strip() for s in paragraph.split('. ') if s.strip()]
                if not sentences:
                    continue
                
                # 슬라이딩 윈도우로 청크 생성
                current_chunk = []
                current_size = 0
                
                for sentence in sentences:
                    sentence_size = len(sentence)
                    
                    # 현재 청크가 비어있으면 무조건 추가
                    if not current_chunk:
                        current_chunk.append(sentence)
                        current_size = sentence_size
                        continue
                    
                    # 현재 청크에 문장을 추가할 수 있는지 확인
                    if current_size + sentence_size <= max_chunk_size:
                        current_chunk.append(sentence)
                        current_size += sentence_size
                    else:
                        # 현재 청크가 최소 크기를 만족하면 저장
                        if current_size >= min_chunk_size:
                            chunk_text = '. '.join(current_chunk) + '.'
                            chunks.append(chunk_text)
                        
                        # 오버랩을 위해 마지막 문장들 유지
                        overlap_sentences = current_chunk[-2:] if len(current_chunk) >= 2 else current_chunk
                        current_chunk = overlap_sentences + [sentence]
                        current_size = sum(len(s) for s in current_chunk)
                
                # 마지막 청크 처리
                if current_chunk and current_size >= min_chunk_size:
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunks.append(chunk_text)
        
        # 청크 품질 검사
        chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_size]
        chunks = [chunk for chunk in chunks if not any(c.isupper() for c in chunk if c.isalpha())]  # 대문자만 있는 청크 제거
        
        print(f"생성된 청크 수: {len(chunks)}")
        print(f"평균 청크 길이: {sum(len(chunk) for chunk in chunks) / len(chunks):.1f} 문자")
        
        # 임베딩 생성 (배치 처리)
        batch_size = 32
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = embedding_model.encode(batch, show_progress_bar=True)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings)
        
        # FAISS 인덱스 생성 (청크 수에 따라 동적으로 설정)
        dimension = embeddings.shape[1]
        nlist = min(4096, max(embeddings.shape[0] // 100, 1))  # 클러스터 수 동적 설정
        
        if len(chunks) < 39:  # 청크가 부족한 경우
            print("경고: 청크 수가 부족합니다. IndexFlatIP를 사용합니다.")
            index = faiss.IndexFlatIP(dimension)  # 단순 코사인 유사도 사용
        else:
            print(f"IndexIVFFlat 사용 (클러스터 수: {nlist})")
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(embeddings.astype('float32'))
        
        index.add(embeddings.astype('float32'))
        
        # 저장
        faiss.write_index(index, os.path.join(VECTOR_DB_PATH, "faiss.index"))
        with open(os.path.join(VECTOR_DB_PATH, "chunks.pkl"), 'wb') as f:
            pickle.dump(chunks, f)
            
        return True
        
    except Exception as e:
        print(f"벡터 DB 생성 중 오류 발생: {str(e)}")
        return False

def load_models():
    """모든 필요한 모델을 로드합니다."""
    print("모델 로드 중...")
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")
    
    # QA 모델 로드 (로컬 경로 사용)
    qa_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR,
        local_files_only=True
    ).to(device)
    
    qa_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        local_files_only=True
    )
    
    # LoRA 가중치 로드
    qa_model = PeftModel.from_pretrained(qa_model, LORA_WEIGHTS_PATH)
    
    # 번역 모델 로드 (원본 MBART 모델 사용)
    from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
    
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

def find_relevant_context(question, vector_db=None, embedding_model=None):
    """질문과 관련된 컨텍스트를 찾습니다."""
    try:
        # 벡터 DB 파일 경로
        vector_db_path = os.path.join(VECTOR_DB_PATH, "faiss.index")
        chunks_path = os.path.join(VECTOR_DB_PATH, "chunks.pkl")
        
        # 벡터 DB가 없거나 접근 권한이 없는 경우 기본 컨텍스트 반환
        if not os.path.exists(vector_db_path) or not os.path.exists(chunks_path):
            print("벡터 DB 파일이 없습니다. 기본 컨텍스트를 사용합니다.")
            return get_default_context(question)
            
        try:
            # 벡터 DB 로드
            vector_db = faiss.read_index(vector_db_path)
            
            # 청크 로드
            with open(chunks_path, 'rb') as f:
                chunks = pickle.load(f)
                
            # 임베딩 모델 로드
            if embedding_model is None:
                embedding_model = SentenceTransformer('intfloat/multilingual-e5-large')
            
            # 질문 임베딩
            question_embedding = embedding_model.encode([question])[0]
            
            # 가장 관련성 높은 컨텍스트 검색
            k = 5  # 상위 5개 검색
            score_threshold = 0.7  # 유사도 임계값 상향 조정
            min_chunks = 2  # 최소 필요 청크 수
            
            # 검색 수행
            distances, indices = vector_db.search(question_embedding.reshape(1, -1).astype('float32'), k)
            
            # 검색된 컨텍스트 반환
            relevant_chunks = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx != -1 and idx < len(chunks):
                    # IP(내적) 기반이므로 distance가 곧 유사도
                    similarity_score = distance
                    if similarity_score >= score_threshold:
                        relevant_chunks.append((chunks[idx], similarity_score))
            
            print(f"검색된 청크 수: {len(relevant_chunks)}")
            if relevant_chunks:
                print(f"최고 유사도 점수: {relevant_chunks[0][1]:.4f}")
            
            # 충분한 관련 컨텍스트가 없거나 유사도가 낮은 경우 기본 컨텍스트 사용
            if len(relevant_chunks) < min_chunks or (relevant_chunks and relevant_chunks[0][1] < score_threshold):
                print(f"충분한 관련 컨텍스트 없음 (청크 수: {len(relevant_chunks)}) → 기본 컨텍스트 사용")
                return get_default_context(question)
            
            # 유사도 점수로 정렬
            relevant_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 2-3개 청크 연결
            top_chunks = relevant_chunks[:3]
            combined_context = "\n".join(chunk for chunk, _ in top_chunks)
            print(f"사용된 청크 수: {len(top_chunks)}")
            
            return combined_context
                
        except Exception as e:
            print(f"벡터 DB 검색 중 오류 발생: {str(e)}")
            return get_default_context(question)
            
    except Exception as e:
        print(f"컨텍스트 검색 중 오류 발생: {str(e)}")
        return get_default_context(question)

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

def translate_text(text, src_lang="ko", tgt_lang="en"):
    """텍스트를 번역합니다."""
    try:
        # 입력 텍스트 정리
        text = clean_text(text)
        
        # 소스 언어 설정 (ko -> ko_KR, en -> en_XX)
        src_lang_code = "ko_KR" if src_lang == "ko" else "en_XX"
        tgt_lang_code = "en_XX" if tgt_lang == "en" else "ko_KR"
        
        # 디버깅용 로그 추가
        print("\n[번역 설정 디버깅]")
        print(f"소스 언어: {src_lang_code}, 목표 언어: {tgt_lang_code}")
        print(f"lang_code_to_id 존재 여부: {'lang_code_to_id' in dir(models['translation_tokenizer'])}")
        
        # 토크나이저 설정
        models['translation_tokenizer'].src_lang = src_lang_code
        
        # 목표 언어의 토큰 ID 가져오기
        try:
            lang_id = models['translation_tokenizer'].lang_code_to_id[tgt_lang_code]
            print(f"목표 lang_id: {lang_id}")
        except (KeyError, AttributeError) as e:
            print(f"언어 코드 {tgt_lang_code}에 대한 토큰 ID를 찾을 수 없습니다: {str(e)}")
            return text
            
        # 토크나이징
        encoded = models['translation_tokenizer'](
            text, 
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(models['device'])
        
        # 번역 생성
        generated_tokens = models['translation_model'].generate(
            **encoded,
            max_length=512,
            num_beams=5,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True,
            forced_bos_token_id=lang_id  # 목표 언어 토큰 ID 설정
        )
        
        # 번역 결과 디코딩 및 정리
        translated = models['translation_tokenizer'].batch_decode(
            generated_tokens, 
            skip_special_tokens=True
        )[0]
        
        # 번역 결과가 비어있거나 원본과 같은 경우
        if not translated or translated == text:
            print(f"번역 실패: 소스 언어={src_lang_code}, 목표 언어={tgt_lang_code}")
            return text
        
        return clean_text(translated)
        
    except Exception as e:
        print(f"번역 중 오류 발생: {str(e)}")
        return text

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
            context = find_relevant_context(question)
            if context:
                english_context = translate_text(context, "ko", "en")
                print(f"\n컨텍스트: {english_context}")
        
        # 3. 프롬프트 구성
        print("\n[3. 프롬프트 구성]")
        if context:
            input_text = f"""Context: {english_context}

Question: {english_question}

Please provide a concise and accurate answer based on the context above. If the context doesn't contain relevant information, say "I don't have enough information to answer that question."

Answer:"""
        else:
            input_text = f"""Question: {english_question}

Please provide a concise and accurate answer.

Answer:"""
        
        print("\n[모델 입력 텍스트]")
        print(input_text)
        
        # 4. 토큰화 및 생성
        print("\n[4. 답변 생성]")
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=512,
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
        
        print(f"\n영어 답변: {english_answer}")
        
        # 5. 영어 답변을 한국어로 번역
        print("\n[5. 답변 번역]")
        
        # 기본 답변 처리
        if "don't have enough information" in english_answer.lower():
            korean_answer = "죄송합니다. 해당 질문에 대한 정보가 충분하지 않습니다."
        else:
            korean_answer = translate_text(english_answer, "en", "ko")
            
            # 번역 결과가 비어있거나 너무 짧은 경우 기본 답변 사용
            if not korean_answer or len(korean_answer) < 2:
                print("번역 결과가 비어있습니다. 기본 답변을 사용합니다.")
                korean_answer = get_default_context(question)
        
        print(f"한국어 답변: {korean_answer}")
        
        return korean_answer
        
    except Exception as e:
        print(f"답변 생성 중 오류 발생: {str(e)}")
        return "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."

def main():
    """메인 함수"""
    try:
        global models
        models = load_models()
        
        # 포트폴리오 데이터 로드 및 벡터 데이터베이스 생성
        print("\n포트폴리오 데이터 로드 중...")
        portfolio_texts = load_portfolio_data()
        if portfolio_texts:
            create_vector_database(portfolio_texts)
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
            
            # 답변 생성
            answer = get_answer(question, models['qa_model'], models['qa_tokenizer'], models['device'], use_context=True)
            print(f"A: {answer}")
            print("-" * 50)
        
        print("\n=== 테스트 완료 ===")
            
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 