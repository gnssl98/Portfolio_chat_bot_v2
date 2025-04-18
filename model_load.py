import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 환경 변수 로드
load_dotenv()

# 모델 경로
MODEL_PATH = r"D:\dataset\fine_tuned_model\flan-t5-large"
OUTPUT_DIR = r"D:\dataset\fine_tuned_model\flan-t5-large-finetuned"

def load_models():
    """모델과 토크나이저를 로드합니다."""
    try:
        print("1. QA 모델 로드 중...")
        qa_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-large",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=False
        )
        print("✓ QA 모델 로드 완료")
        
        print("\n2. QA 토크나이저 로드 중...")
        qa_tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-large",
            local_files_only=False
        )
        print("✓ QA 토크나이저 로드 완료")
        
        print("\n3. 번역 모델 로드 중...")
        translation_model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-large",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            local_files_only=False
        )
        print("✓ 번역 모델 로드 완료")
        
        print("\n4. 번역 토크나이저 로드 중...")
        translation_tokenizer = AutoTokenizer.from_pretrained(
            "google/flan-t5-large",
            local_files_only=False
        )
        print("✓ 번역 토크나이저 로드 완료")
        
        # 모델 저장
        print("\n5. 모델 저장 중...")
        qa_model.save_pretrained(MODEL_PATH)
        qa_tokenizer.save_pretrained(MODEL_PATH)
        print("✓ 모델 저장 완료")
        
        return qa_model, qa_tokenizer, translation_model, translation_tokenizer
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        raise e

if __name__ == "__main__":
    try:
        qa_model, qa_tokenizer, translation_model, translation_tokenizer = load_models()
        print("\n모델 로드 완료!")
    except Exception as e:
        print(f"\n프로그램 종료 (오류 발생): {str(e)}") 