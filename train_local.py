import os
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    MBartForConditionalGeneration,
    MBart50TokenizerFast
)
import torch
from tqdm import tqdm
import numpy as np
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
import PyPDF2
import glob
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import json

# 환경 변수 로드
load_dotenv()

# PyTorch 메모리 관리 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 경로 설정
MODEL_ID = "google/flan-t5-large"  # 더 작은 모델로 변경
OUTPUT_DIR = r"D:\dataset\fine_tuned_model\flan-t5-large-finetuned"
CACHE_DIR = r"D:\dataset\huggingface_cache"
PDF_DIR = "./portfolio_data"
TXT_DIR = "./portfolio_data"
VECTOR_DB_PATH = "./vector_db"
TRANSLATION_MODEL_PATH = "./translation_model"
VENV_PATH = r"D:\dataset\portfolio_chatbot_env"

# LoRA 설정
LORA_R = 16  # LoRA의 랭크
LORA_ALPHA = 32  # LoRA의 알파 값
LORA_DROPOUT = 0.1  # LoRA의 드롭아웃 비율
LORA_TARGET_MODULES = ["q", "v"]  # LoRA를 적용할 모듈

def extract_text_from_pdf(pdf_path):
    """PDF 파일에서 텍스트를 추출합니다."""
    print(f"\nPDF 파일 처리 중: {pdf_path}")
    text = ""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for i, page in enumerate(pdf_reader.pages, 1):
            print(f"페이지 {i}/{len(pdf_reader.pages)} 처리 중...")
            text += page.extract_text() + "\n"
    return text

def load_text_files():
    """PDF와 txt 파일에서 텍스트를 로드합니다."""
    texts = []
    
    # PDF 파일 처리 (하위 폴더 포함)
    pdf_files = glob.glob(os.path.join(PDF_DIR, "**", "*.pdf"), recursive=True)
    print(f"\n발견된 PDF 파일: {pdf_files}")
    for pdf_file in pdf_files:
        texts.append(extract_text_from_pdf(pdf_file))
    
    # txt 파일 처리 (하위 폴더 포함)
    txt_files = glob.glob(os.path.join(TXT_DIR, "**", "*.txt"), recursive=True)
    print(f"\n발견된 TXT 파일: {txt_files}")
    for txt_file in txt_files:
        print(f"\nTXT 파일 처리 중: {txt_file}")
        with open(txt_file, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    
    if not texts:
        print(f"경고: {PDF_DIR}에서 PDF나 txt 파일을 찾을 수 없습니다.")
        print(f"현재 디렉토리: {os.getcwd()}")
        print(f"디렉토리 내용: {os.listdir(PDF_DIR)}")
        for root, dirs, files in os.walk(PDF_DIR):
            print(f"\n{root} 폴더 내용:")
            for file in files:
                print(f"- {file}")
    
    return texts

def create_vector_database(texts, chunk_size=500):
    """텍스트를 청크로 나누고 벡터 데이터베이스를 생성합니다."""
    print("벡터 데이터베이스 생성 중...")
    
    # 문장 임베딩 모델 로드
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 텍스트를 청크로 분할
    chunks = []
    for text in texts:
        # 텍스트를 문장 단위로 분할
        sentences = text.split('. ')
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
    
    # 청크 임베딩 생성
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    # FAISS 인덱스 생성
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    # 벡터 DB 저장
    os.makedirs(VECTOR_DB_PATH, exist_ok=True)
    faiss.write_index(index, os.path.join(VECTOR_DB_PATH, "faiss.index"))
    with open(os.path.join(VECTOR_DB_PATH, "chunks.pkl"), 'wb') as f:
        pickle.dump(chunks, f)
    
    print(f"✓ 벡터 데이터베이스 생성 완료 (총 {len(chunks)}개 청크)")
    return index, chunks

def load_translation_model():
    """번역 모델을 로드합니다."""
    print("번역 모델 로드 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
    print("✓ 번역 모델 로드 완료")
    return model, tokenizer, device

def translate_text(text, model, tokenizer, device, src_lang="ko_KR", tgt_lang="en_XX"):
    """텍스트를 번역합니다."""
    tokenizer.src_lang = src_lang
    encoded = tokenizer(text, return_tensors="pt").to(device)
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
        max_length=512,
        num_beams=5,
        early_stopping=True
    )
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

def create_qa_pairs(texts, translation_model, translation_tokenizer, translation_device):
    """텍스트에서 질문-답변 쌍을 생성합니다."""
    # 자기소개 관련 기본 질문들 (한국어)
    base_questions = [
        "당신의 이름은 무엇인가요?",
        "당신의 직업은 무엇인가요?",
        "당신의 학력은 어떻게 되나요?",
        "당신의 경력은 어떻게 되나요?",
        "당신의 기술 스택은 무엇인가요?",
        "당신의 프로젝트 경험을 설명해주세요.",
        "당신의 취미는 무엇인가요?",
        "당신의 목표는 무엇인가요?",
        "당신의 강점은 무엇인가요?",
        "당신의 약점은 무엇인가요?",
        # 추가 질문들
        "어떤 프로그래밍 언어를 사용할 수 있나요?",
        "가장 자신있는 기술은 무엇인가요?",
        "최근에 진행한 프로젝트는 무엇인가요?",
        "팀 프로젝트 경험이 있나요?",
        "어떤 개발 도구들을 사용해보셨나요?",
        "영어 실력은 어느 정도인가요?",
        "왜 개발자가 되고 싶으신가요?",
        "앞으로의 커리어 계획은 무엇인가요?",
        "개발자로서의 철학은 무엇인가요?",
        "문제 해결 능력은 어떠신가요?"
    ]
    
    # 영어 질문들
    english_questions = [
        "What is your name?",
        "What is your occupation?",
        "What is your educational background?",
        "What is your work experience?",
        "What is your technical stack?",
        "Please describe your project experience.",
        "What are your hobbies?",
        "What are your goals?",
        "What are your strengths?",
        "What are your weaknesses?",
        # 추가 질문들의 영어 버전
        "Which programming languages can you use?",
        "What is your strongest technical skill?",
        "What was your most recent project?",
        "Do you have team project experience?",
        "What development tools have you used?",
        "How proficient are you in English?",
        "Why do you want to be a developer?",
        "What are your future career plans?",
        "What is your philosophy as a developer?",
        "How would you describe your problem-solving skills?"
    ]
    
    qa_pairs = []
    for ko_q, en_q in zip(base_questions, english_questions):
        # 질문에 대한 관련 컨텍스트 찾기
        relevant_context = find_relevant_context(ko_q, texts)
        
        # 컨텍스트를 영어로 번역
        translated_context = translate_text(
            relevant_context,
            translation_model,
            translation_tokenizer,
            translation_device
        )
        
        # 개선된 프롬프트 형식 사용
        input_text = f"""You are a professional AI assistant. Your task is to provide accurate and relevant answers based on the given context.

Context: {translated_context}

Question: {en_q}

Instructions:
1. Use only information from the provided context
2. If the context doesn't contain relevant information, respond with "I don't have enough information to answer that question."
3. Keep your answer concise and focused
4. Maintain a professional tone
5. If multiple pieces of information are relevant, combine them coherently

Answer:"""
        
        # 답변 생성
        answer = translated_context  # 실제 구현에서는 여기서 모델을 통해 답변을 생성해야 함
        
        qa_pairs.append({
            "question": input_text,
            "context": translated_context,
            "answer": answer
        })
    
    return qa_pairs

def find_relevant_context(question, texts, top_k=3):
    """질문과 관련된 컨텍스트를 찾습니다."""
    # 문장 임베딩 모델 로드
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    # 질문 임베딩 생성
    question_embedding = model.encode([question])[0]
    
    # FAISS 인덱스 로드
    index = faiss.read_index(os.path.join(VECTOR_DB_PATH, "faiss.index"))
    
    # 가장 관련성 높은 청크 검색
    D, I = index.search(np.array([question_embedding]).astype('float32'), top_k)
    
    # 청크 로드
    with open(os.path.join(VECTOR_DB_PATH, "chunks.pkl"), 'rb') as f:
        chunks = pickle.load(f)
    
    # 관련 컨텍스트 결합
    relevant_chunks = [chunks[i] for i in I[0]]
    return " ".join(relevant_chunks)

def prepare_dataset():
    """데이터셋을 준비합니다."""
    print("데이터셋 준비 중...")
    
    # 번역 모델 로드
    translation_model, translation_tokenizer, translation_device = load_translation_model()
    
    # 텍스트 로드
    texts = load_text_files()
    if not texts:
        raise ValueError("PDF나 txt 파일을 찾을 수 없습니다.")
    
    # 벡터 데이터베이스 생성
    create_vector_database(texts)
    
    # QA 쌍 생성
    qa_pairs = create_qa_pairs(texts, translation_model, translation_tokenizer, translation_device)
    
    # 데이터셋 생성
    dataset = Dataset.from_dict({
        "question": [pair["question"] for pair in qa_pairs],
        "context": [pair["context"] for pair in qa_pairs],
        "answers": [{"text": [pair["answer"]]} for pair in qa_pairs]
    })
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR
    )
    
    def preprocess_function(examples):
        inputs = [f"question: {q}\ncontext: {c}" for q, c in zip(examples["question"], examples["context"])]
        targets = [ans["text"][0] if ans["text"] else "" for ans in examples["answers"]]
        
        model_inputs = tokenizer(
            inputs,
            max_length=512,
            padding="max_length",
            truncation=True
        )
        
        labels = tokenizer(
            targets,
            max_length=128,
            padding="max_length",
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # 데이터셋 전처리
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 학습/검증 분할
    tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    tokenized_dataset["validation"] = tokenized_dataset["test"]
    del tokenized_dataset["test"]
    
    return tokenized_dataset

def train():
    """모델을 LoRA로 파인튜닝합니다."""
    try:
        print("LoRA 파인튜닝 시작...")
        
        # GPU 설정
        print("GPU 상태 확인 중...")
        print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA 버전: {torch.version.cuda}")
            print(f"사용 가능한 GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU 개수: {torch.cuda.device_count()}")
            device = torch.device("cuda")
        else:
            print("GPU를 사용할 수 없습니다. CPU 모드로 실행합니다.")
            device = torch.device("cpu")
        
        # 데이터셋 준비
        print("1. 데이터셋 준비 중...")
        tokenized_dataset = prepare_dataset()
        print("✓ 데이터셋 준비 완료")
        
        # 토크나이저 로드
        print("2. 토크나이저 로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR
        )
        print("✓ 토크나이저 로드 완료")
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"GPU 메모리 상태: {torch.cuda.memory_allocated()/1024**2:.2f}MB / {torch.cuda.memory_reserved()/1024**2:.2f}MB")
        
        # 모델 로드
        print("3. 기본 모델 로드 중...")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,  # FP16 사용
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR
        )
        
        # LoRA 설정
        print("4. LoRA 설정 중...")
        model = prepare_model_for_kbit_training(model)
        
        # LoRA 설정
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM
        )
        
        # LoRA 모델 생성
        model = get_peft_model(model, lora_config)
        model = model.to(device)
        print("✓ LoRA 모델 설정 완료")
        
        # 학습 인자 설정
        print("5. 학습 인자 설정 중...")
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="steps",
            eval_steps=100,
            learning_rate=1e-4,
            per_device_train_batch_size=4,  # 배치 크기 증가
            per_device_eval_batch_size=4,
            num_train_epochs=5,
            weight_decay=0.05,
            save_total_limit=3,
            predict_with_generate=True,
            fp16=True,  # FP16 활성화
            report_to="none",
            gradient_accumulation_steps=8,  # 그래디언트 누적 스텝 감소
            dataloader_pin_memory=True,  # GPU 학습을 위해 pin_memory 활성화
            no_cuda=False,  # GPU 사용 활성화
            warmup_steps=100,
            logging_steps=50,
            save_steps=200,
            max_grad_norm=1.0,
            optim="adamw_torch",
            gradient_checkpointing=True
        )
        print("✓ 학습 인자 설정 완료")
        
        # 트레이너 초기화
        print("6. 트레이너 초기화 중...")
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True
            )
        )
        print("✓ 트레이너 초기화 완료")
        
        # 학습 시작
        print("\n=== LoRA 학습 시작 ===")
        trainer.train()
        
        # LoRA 가중치 저장
        print("\nLoRA 가중치 저장 중...")
        model.save_pretrained(OUTPUT_DIR)
        print("✓ LoRA 가중치 저장 완료")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        # QA 모델 학습
        train()
    except Exception as e:
        print(f"\n프로그램 종료 (오류 발생): {str(e)}") 