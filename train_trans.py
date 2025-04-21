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
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 환경 변수 설정
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
OUTPUT_DIR = "./translation_model"  # 기본 저장 경로
SAVE_DIR_KO_EN = r"D:\dataset\fine_tuned_model\translation_model_finetunned_ko_en"    # 한국어->영어 모델 저장 경로
SAVE_DIR_EN_KO = r"D:\dataset\fine_tuned_model\translation_model_finetunned_en_ko"    # 영어->한국어 모델 저장 경로
BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 8
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
MAX_LENGTH = 256

def create_translation_dataset(direction="ko_en"):
    """번역 데이터셋을 생성합니다."""
    # 예시 데이터 (실제로는 더 많은 데이터가 필요합니다)
    data = {
        "ko": [
            "ADsP (데이터 분석 준전문가) 자격증을 2024년 11월에 취득했습니다.",
            "네트워크 관리사 2급 자격증을 2019년 1월에 취득했습니다.",
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
            "I obtained the ADsP (Advanced Data Analytics Semi-Professional) certification in November 2024.",
            "I obtained the Level 2 Network Administrator certification in January 2019.",
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
    
    # 데이터셋 생성
    train_dataset = []
    
    if direction == "ko_en":
        for ko, en in zip(data["ko"], data["en"]):
            train_dataset.append({
                "translation": {
                    "ko": ko,
                    "en": en
                }
            })
    else:  # en_ko
        for ko, en in zip(data["ko"], data["en"]):
            train_dataset.append({
                "translation": {
                    "en": en,
                    "ko": ko
                }
            })
    
    # 검증 데이터셋은 2개 항목 사용
    val_dataset = train_dataset[:2]
    
    return {"train": train_dataset, "validation": val_dataset}

def preprocess_function(examples, tokenizer, direction="ko_en", max_length=MAX_LENGTH):
    """데이터 전처리 함수"""
    if direction == "ko_en":
        src_lang = "ko_KR"
        tgt_lang = "en_XX"
    else:  # en_ko
        src_lang = "en_XX"
        tgt_lang = "ko_KR"
    
    # 명시적으로 src_lang 설정
    tokenizer.src_lang = src_lang
    
    # 소스 텍스트와 타겟 텍스트 준비
    source_texts = [example["translation"][src_lang] for example in examples]
    target_texts = [example["translation"][tgt_lang] for example in examples]
    
    # 입력 텍스트 토큰화
    model_inputs = tokenizer(
        source_texts,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # 타겟 텍스트 토큰화를 위해 tgt_lang 설정
    tokenizer.tgt_lang = tgt_lang
    
    # 타겟 텍스트 토큰화
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target_texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
    
    model_inputs["labels"] = labels["input_ids"]
    
    # 디버깅을 위한 정보 출력
    logger.info(f"\n[전처리 정보]")
    logger.info(f"Source language: {src_lang}")
    logger.info(f"Target language: {tgt_lang}")
    logger.info(f"입력 텍스트 예시: {source_texts[0]}")
    logger.info(f"목표 텍스트 예시: {target_texts[0]}")
    logger.info(f"입력 토큰 수: {model_inputs['input_ids'].shape}")
    logger.info(f"라벨 토큰 수: {model_inputs['labels'].shape}")
    
    return {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "labels": model_inputs["labels"]
    }

class TranslationDataset(Dataset):
    def __init__(self, examples, tokenizer, direction="ko_en"):
        self.examples = examples
        self.tokenizer = tokenizer
        self.direction = direction
        
        # 데이터 전처리
        processed = preprocess_function(examples, tokenizer, direction)
        self.input_ids = processed["input_ids"]
        self.attention_mask = processed["attention_mask"]
        self.labels = processed["labels"]
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx]
        }

def translate_text(text: str, model, tokenizer, direction="ko_en") -> str:
    """텍스트를 번역합니다."""
    try:
        # 입력 텍스트가 비어있는 경우
        if not text or not text.strip():
            return text
        
        # 방향에 따른 언어 코드 설정
        if direction == "ko_en":
            src_lang = "ko_KR"
            tgt_lang = "en_XX"
        else:  # en_ko
            src_lang = "en_XX"
            tgt_lang = "ko_KR"
        
        # 명시적으로 src_lang 설정
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        
        # forced_bos_token_id 설정
        forced_bos_token_id = tokenizer.lang_code_to_id[tgt_lang]
        
        # 디버깅 정보 출력
        logger.info(f"\n[번역 설정]")
        logger.info(f"원본 텍스트: {text[:100]}...")
        logger.info(f"번역 방향: {direction}")
        logger.info(f"원본 언어(src_lang): {src_lang}")
        logger.info(f"목표 언어(tgt_lang): {tgt_lang}")
        logger.info(f"forced_bos_token_id: {forced_bos_token_id}")
        
        # 텍스트 토큰화
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512,
            padding=True
        )
        
        # 입력을 GPU로 이동
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 번역 생성
        with torch.no_grad():
            translated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=5,
                length_penalty=1.0,
                no_repeat_ngram_size=2,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        # 번역 결과 디코딩
        result = tokenizer.decode(translated[0], skip_special_tokens=True)
        
        logger.info(f"번역 결과: {result[:100]}...")
        
        return result
        
    except Exception as e:
        logger.error(f"번역 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return text  # 오류 발생 시 원본 텍스트 반환

def is_korean(text: str) -> bool:
    """텍스트가 한국어인지 확인합니다."""
    if not text:
        return False
    # 한글 유니코드 범위: AC00-D7A3 (가-힣)
    korean_char_count = len([c for c in text if '\uAC00' <= c <= '\uD7A3'])
    return korean_char_count > len(text) * 0.3  # 30% 이상이 한글이면 한국어로 간주

def save_model(model, tokenizer, save_dir):
    """학습된 모델과 토크나이저를 저장합니다."""
    try:
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # 현재 언어 설정 저장
        config = {
            "src_lang": tokenizer.src_lang,
            "tgt_lang": tokenizer.tgt_lang
        }
        
        # 설정 파일 저장
        with open(os.path.join(save_dir, "language_config.json"), "w") as f:
            json.dump(config, f)
        
        # 모델 저장
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        logger.info(f"모델이 성공적으로 저장되었습니다: {save_dir}")
        logger.info(f"언어 설정: {config}")
        
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
        # 방향에 따른 설정
        if direction == "ko_en":
            src_lang = "ko_KR"
            tgt_lang = "en_XX"
            save_dir = SAVE_DIR_KO_EN
            logger.info("한국어 -> 영어 모델 학습을 시작합니다.")
        else:  # en_ko
            src_lang = "en_XX"
            tgt_lang = "ko_KR"
            save_dir = SAVE_DIR_EN_KO
            logger.info("영어 -> 한국어 모델 학습을 시작합니다.")
            
        logger.info(f"Source Language: {src_lang}")
        logger.info(f"Target Language: {tgt_lang}")
        logger.info(f"저장 경로: {save_dir}")

        # CUDA 사용 가능 여부 확인
        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Please check your GPU installation.")
            return
            
        device = torch.device("cuda")
        logger.info(f"Using device: {device}")
        
        # CUDA 메모리 캐시 초기화
        torch.cuda.empty_cache()
        
        # 모델과 토크나이저 로드
        logger.info("모델과 토크나이저를 로드합니다...")
        model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)
        tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
        
        # 언어 코드 설정 검증
        logger.info("언어 코드 설정을 검증합니다...")
        if src_lang not in tokenizer.lang_code_to_id:
            raise ValueError(f"Invalid source language code: {src_lang}")
        if tgt_lang not in tokenizer.lang_code_to_id:
            raise ValueError(f"Invalid target language code: {tgt_lang}")
            
        # 토크나이저 설정
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang
        
        # 설정 검증을 위한 테스트 번역
        test_text = "테스트 문장입니다." if direction == "ko_en" else "This is a test sentence."
        logger.info(f"번역 설정 테스트를 수행합니다.")
        logger.info(f"테스트 입력: {test_text}")
        
        test_inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            test_outputs = model.generate(
                **test_inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                max_length=512,
                num_beams=5
            )
        test_translation = tokenizer.decode(test_outputs[0], skip_special_tokens=True)
        logger.info(f"테스트 번역 결과: {test_translation}")
        
        # 데이터셋 생성
        logger.info("데이터셋을 생성합니다...")
        raw_datasets = create_translation_dataset(direction=direction)
        
        # 데이터 콜레이터 설정
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            return_tensors="pt"
        )
        
        def preprocess_function(examples):
            if direction == "ko_en":
                source_lang = "ko"
                target_lang = "en"
            else:
                source_lang = "en"
                target_lang = "ko"
                
            inputs = [ex["translation"][source_lang] for ex in examples]
            targets = [ex["translation"][target_lang] for ex in examples]
            
            # 토크나이저 설정
            tokenizer.src_lang = "ko_KR" if source_lang == "ko" else "en_XX"
            model_inputs = tokenizer(
                inputs, 
                max_length=MAX_LENGTH, 
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # 타겟 설정
            tokenizer.tgt_lang = "en_XX" if target_lang == "en" else "ko_KR"
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(
                    targets,
                    max_length=MAX_LENGTH,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
            
            model_inputs["labels"] = labels["input_ids"]
            
            # 텐서를 numpy 배열로 변환
            return {
                "input_ids": model_inputs["input_ids"].numpy(),
                "attention_mask": model_inputs["attention_mask"].numpy(),
                "labels": model_inputs["labels"].numpy()
            }
        
        # 데이터셋 전처리 - 배치로 처리
        processed_datasets = {}
        for key, dataset in raw_datasets.items():
            # 배치 크기만큼 데이터를 모아서 한 번에 처리
            batch_size = 8
            processed_data = []
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i + batch_size]
                processed = preprocess_function(batch)
                processed_data.extend([{
                    "input_ids": processed["input_ids"][j],
                    "attention_mask": processed["attention_mask"][j],
                    "labels": processed["labels"][j]
                } for j in range(len(batch))])
            processed_datasets[key] = processed_data
        
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"] if "validation" in processed_datasets else None
        
        # 데이터셋 정보 출력
        logger.info(f"\n[데이터셋 정보]")
        logger.info(f"학습 데이터 크기: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"검증 데이터 크기: {len(eval_dataset)}")
        
        # 샘플 데이터 형태 확인
        sample = train_dataset[0]
        logger.info(f"\n[샘플 데이터 형태]")
        logger.info(f"input_ids shape: {sample['input_ids'].shape}")
        logger.info(f"attention_mask shape: {sample['attention_mask'].shape}")
        logger.info(f"labels shape: {sample['labels'].shape}")
        
        # 학습 설정
        training_args = Seq2SeqTrainingArguments(
            output_dir=save_dir,
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            save_strategy="steps",
            eval_strategy="steps",
            save_steps=100,
            eval_steps=100,
            logging_dir=os.path.join(save_dir, "logs"),
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            do_eval=True,
            fp16=True,
            gradient_checkpointing=True,
            per_device_eval_batch_size=BATCH_SIZE,
            predict_with_generate=True,
            generation_max_length=MAX_LENGTH,
        )
        
        # 트레이너 설정
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        # 모델 메모리 최적화 설정
        model.gradient_checkpointing_enable()
        
        # CUDA 캐시 초기화
        torch.cuda.empty_cache()
        
        # 메모리 상태 출력
        logger.info(f"GPU 메모리 상태:")
        logger.info(f"할당된 메모리: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"캐시된 메모리: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        # 학습 시작
        logger.info("모델 학습을 시작합니다...")
        trainer.train()
        
        # 모델 저장 전 최종 설정 검증
        logger.info("최종 설정을 검증합니다...")
        logger.info(f"Source Language: {tokenizer.src_lang}")
        logger.info(f"Target Language: {tokenizer.tgt_lang}")
        
        # 모델과 토크나이저 저장
        logger.info(f"모델을 저장합니다: {save_dir}")
        save_model(model, tokenizer, save_dir)
        
        # 저장된 모델 검증
        logger.info("저장된 모델을 검증합니다...")
        loaded_tokenizer = MBart50TokenizerFast.from_pretrained(save_dir)
        loaded_model = MBartForConditionalGeneration.from_pretrained(save_dir)
        
        # 저장된 모델로 테스트 번역 수행
        loaded_tokenizer.src_lang = src_lang
        loaded_tokenizer.tgt_lang = tgt_lang
        test_inputs = loaded_tokenizer(test_text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            test_outputs = loaded_model.generate(
                **test_inputs,
                forced_bos_token_id=loaded_tokenizer.lang_code_to_id[tgt_lang],
                max_length=512,
                num_beams=5
            )
        final_test_translation = loaded_tokenizer.decode(test_outputs[0], skip_special_tokens=True)
        logger.info(f"최종 테스트 번역 결과: {final_test_translation}")
        
        logger.info("학습이 완료되었습니다!")
        
    except Exception as e:
        logger.error(f"학습 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    # 한국어->영어 모델 학습
    logger.info("Training Korean->English model...")
    train("ko_en")
    
    # 영어->한국어 모델 학습
    logger.info("Training English->Korean model...")
    train("en_ko") 