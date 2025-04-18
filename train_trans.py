import os
import torch
from transformers import (
    MBartForConditionalGeneration,
    MBart50TokenizerFast,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset
import numpy as np
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import pipeline

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 설정
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = "./translation_model"  # 기본 저장 경로
SAVE_DIR_KO_EN = r"D:\dataset\fine_tuned_model\translation_model_finetunned_ko_en"    # 한국어->영어 모델 저장 경로
SAVE_DIR_EN_KO = r"D:\dataset\fine_tuned_model\translation_model_finetunned_en_ko"    # 영어->한국어 모델 저장 경로
BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_LENGTH = 512

class TranslationDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids.cpu()  # CPU로 이동
        self.attention_mask = attention_mask.cpu()  # CPU로 이동
        self.labels = labels.cpu()  # CPU로 이동
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def create_translation_dataset(direction="ko_en"):
    """번역 데이터셋을 생성합니다."""
    # 예시 데이터 (실제로는 더 많은 데이터가 필요합니다)
    data = {
        "ko": [
            "안녕하세요, 저는 AI 엔지니어입니다.",
            "저는 Python과 JavaScript를 사용하여 웹 애플리케이션을 개발합니다.",
            "머신러닝과 딥러닝에 대한 깊은 이해를 가지고 있습니다.",
            "프로젝트 경험은 3년 이상입니다.",
            "팀 프로젝트에서 리더 역할을 수행한 경험이 있습니다.",
            "문제 해결 능력이 뛰어나며 새로운 기술을 빠르게 배웁니다.",
            "커뮤니케이션 능력이 좋고 팀워크를 중요시합니다.",
            "지속적인 자기 개발과 학습에 관심이 많습니다.",
            "최신 기술 트렌드를 주시하고 있습니다.",
            "코드 품질과 유지보수성을 중요시합니다.",
            "저는 ADsP 자격증과 네트워크 관리사를 취득했습니다.",
            "ADsP 자격증은 데이터 분석과 관련된 전문성을 보여줍니다.",
            "네트워크 관리사 자격증은 시스템 관리 능력을 입증합니다.",
            "자격증 취득을 통해 지속적인 자기 개발을 실천하고 있습니다.",
            "추가적인 자격증 취득을 계획하고 있습니다.",
            # OCR 관련 질문과 대답 추가
            "OCR 기술을 활용한 화장품 성분 분석 프로젝트를 진행했습니다.",
            "한국어 기반 OCR 모델을 개발하여 화장품 성분을 인식했습니다.",
            "화장품 성분 사진에서 텍스트를 추출하는 작업을 수행했습니다.",
            "AI Hub의 의약품, 화장품 패키징 OCR 데이터셋을 활용했습니다.",
            "화장품 성분 사진의 텍스트를 30자 이하로 잘라서 학습했습니다.",
            "OCR 모델이 추출한 텍스트를 GPT Assistant로 정제했습니다.",
            "LMDB를 사용하여 데이터를 효율적으로 관리했습니다.",
            "Naver Clova AI의 deep-text-recognition-benchmark를 참고했습니다.",
            "화장품 성분 기반으로 Score를 계산하는 알고리즘을 개발했습니다.",
            "Cosine 유사도를 활용하여 화장품을 추천하는 시스템을 구축했습니다."
        ],
        "en": [
            "Hello, I am an AI engineer.",
            "I develop web applications using Python and JavaScript.",
            "I have a deep understanding of machine learning and deep learning.",
            "I have over 3 years of project experience.",
            "I have experience in leading team projects.",
            "I have excellent problem-solving skills and learn new technologies quickly.",
            "I have good communication skills and value teamwork.",
            "I am interested in continuous self-development and learning.",
            "I keep an eye on the latest technology trends.",
            "I value code quality and maintainability.",
            "I have obtained the ADsP certification and Network Administrator certification.",
            "The ADsP certification demonstrates expertise in data analysis.",
            "The Network Administrator certification proves my system management capabilities.",
            "I practice continuous self-development through obtaining certifications.",
            "I am planning to obtain additional certifications.",
            # OCR 관련 질문과 대답 추가
            "I developed a project for analyzing cosmetic ingredients using OCR technology.",
            "I developed a Korean-based OCR model for recognizing cosmetic ingredients.",
            "I performed text extraction from cosmetic ingredient images.",
            "I utilized AI Hub's pharmaceutical and cosmetic packaging OCR dataset.",
            "I trained the model by cutting cosmetic ingredient images into text segments under 30 characters.",
            "I refined the text extracted by the OCR model using GPT Assistant.",
            "I used LMDB for efficient data management.",
            "I referenced Naver Clova AI's deep-text-recognition-benchmark.",
            "I developed an algorithm to calculate Scores based on cosmetic ingredients.",
            "I built a system to recommend cosmetics using Cosine similarity."
        ]
    }
    
    # 데이터셋 생성 (단방향)
    train_dataset = []
    
    if direction == "ko_en":
        # 한국어 -> 영어 방향
        for ko, en in zip(data["ko"], data["en"]):
            train_dataset.append({
                "source_text": ko,
                "target_text": en
            })
    else:  # en_ko
        # 영어 -> 한국어 방향
        for ko, en in zip(data["ko"], data["en"]):
            train_dataset.append({
                "source_text": en,
                "target_text": ko
            })
    
    # 검증 데이터셋은 2개 항목 사용
    val_dataset = train_dataset[:2]
    
    return {"train": train_dataset, "validation": val_dataset}

def preprocess_function(examples, tokenizer, direction="ko_en", max_length=MAX_LENGTH):
    """데이터 전처리 함수"""
    # 소스 텍스트와 타겟 텍스트 준비
    source_texts = [example["source_text"] for example in examples]
    target_texts = [example["target_text"] for example in examples]
    
    # 방향에 따라 토크나이저 설정
    if direction == "ko_en":
        tokenizer.src_lang = "ko_KR"
        tokenizer.tgt_lang = "en_XX"
    else:  # en_ko
        tokenizer.src_lang = "en_XX"
        tokenizer.tgt_lang = "ko_KR"
    
    # 입력 텍스트 토크나이징
    model_inputs = tokenizer(
        source_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # 타겟 텍스트 토크나이징
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    
    # 텐서를 CPU로 이동한 후 numpy 배열로 변환
    return {
        "input_ids": model_inputs["input_ids"].cpu().numpy(),
        "attention_mask": model_inputs["attention_mask"].cpu().numpy(),
        "labels": model_inputs["labels"].cpu().numpy()
    }

def translate_text(text, model, tokenizer, src_lang="ko_KR", tgt_lang="en_XX"):
    """텍스트 번역 함수"""
    try:
        if not text.strip():
            return ""
            
        # 파이프라인 생성
        translator = pipeline(
            "translation",
            model=model,
            tokenizer=tokenizer,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            device=0  # GPU 사용
        )
        
        # forced_bos_token_id 설정
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
        
        # 번역 수행
        result = translator(
            text,
            max_length=512,
            forced_bos_token_id=forced_bos_token_id
        )[0]["translation_text"]
        
        return result
        
    except Exception as e:
        logger.error(f"번역 중 오류 발생: {str(e)}")
        return text

def save_model(model, tokenizer, save_dir=SAVE_DIR_KO_EN):
    """학습된 모델과 토크나이저를 저장합니다."""
    try:
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 모델 저장
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        logger.info(f"모델이 성공적으로 저장되었습니다: {save_dir}")
        
        # 저장된 파일 목록 출력
        saved_files = os.listdir(save_dir)
        logger.info("저장된 파일 목록:")
        for file in saved_files:
            logger.info(f"- {file}")
            
    except Exception as e:
        logger.error(f"모델 저장 중 오류 발생: {str(e)}")
        raise

def train(direction="ko_en"):
    """번역 모델 학습 함수"""
    try:
        # CUDA 사용 가능 여부 확인
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Please check your GPU installation.")
            return
            
        # 디바이스 설정
        device = torch.device("cuda")
        logger.info(f"Using device: {device}")
        
        # CUDA 메모리 캐시 초기화
        torch.cuda.empty_cache()
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # 모델과 토크나이저 로드
        logger.info("Loading model and tokenizer...")
        model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
        tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
        
        # 모델을 GPU로 이동
        model = model.to(device)
        
        # LoRA 설정
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"],
            bias="none",
            modules_to_save=["embed_tokens", "shared"]
        )
        
        # LoRA 모델 생성
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
        # 데이터셋 생성
        logger.info("Creating dataset...")
        dataset = create_translation_dataset(direction)
        
        # 데이터 전처리
        logger.info("Preprocessing data...")
        train_features = preprocess_function(dataset["train"], tokenizer, direction)
        eval_features = preprocess_function(dataset["validation"], tokenizer, direction)
        
        # numpy 배열을 텐서로 변환
        train_features = {
            k: torch.tensor(v, dtype=torch.long) 
            for k, v in train_features.items()
        }
        eval_features = {
            k: torch.tensor(v, dtype=torch.long) 
            for k, v in eval_features.items()
        }
        
        # 커스텀 데이터셋 생성
        train_dataset = TranslationDataset(
            train_features["input_ids"],
            train_features["attention_mask"],
            train_features["labels"]
        )
        
        eval_dataset = TranslationDataset(
            eval_features["input_ids"],
            eval_features["attention_mask"],
            eval_features["labels"]
        )
        
        # 데이터 콜레이터 설정
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True
        )
        
        # 학습 인자 설정
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            eval_steps=100,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=0.01,
            save_total_limit=2,
            predict_with_generate=True,
            fp16=True,  # fp16 활성화
            fp16_opt_level="O1",  # fp16 최적화 레벨 설정
            gradient_checkpointing=True,  # 메모리 효율을 위한 그래디언트 체크포인팅 활성화
            logging_dir="./logs",
            logging_steps=10,
            report_to="none",
            label_names=["labels"]  # 레이블 이름 추가
        )
        
        # 트레이너 초기화
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator
        )
        
        # 모델 학습
        logger.info("Starting training...")
        trainer.train()
        
        # 모델 저장
        logger.info("Saving model...")
        save_dir = SAVE_DIR_KO_EN if direction == "ko_en" else SAVE_DIR_EN_KO
        save_model(model, tokenizer, save_dir)
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    # 한국어->영어 모델 학습
    logger.info("Training Korean->English model...")
    train("ko_en")
    
    # 영어->한국어 모델 학습
    logger.info("Training English->Korean model...")
    train("en_ko") 