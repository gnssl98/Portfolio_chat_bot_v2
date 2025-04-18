import os
import sys
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 환경 변수 로드
load_dotenv()

# 경로 설정
MODEL_ID = r"D:\dataset\fine_tuned_model\flan-t5-large"  # 로컬 기본 모델 경로
CACHE_DIR = r"D:\dataset\huggingface_cache"

def main():
    """메인 함수"""
    try:
        # QA 모델 로드
        qa_model = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR
        )
        qa_tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=CACHE_DIR
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qa_model.to(device)

        print("\n=== 질문-답변 시스템 시작 ===")
        print("종료하려면 'exit'를 입력하세요.")

        while True:
            question = input("\n질문을 입력하세요: ")
            if question.lower() == 'exit':
                break

            # 질문을 토큰화
            inputs = qa_tokenizer(question, return_tensors="pt", max_length=512, truncation=True).to(device)

            # 답변 생성
            with torch.no_grad():
                outputs = qa_model.generate(
                    **inputs,
                    max_length=150,
                    min_length=10,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    early_stopping=True
                )

            answer = qa_tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"답변: {answer}")

        print("\n=== 질문-답변 시스템 종료 ===")

    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()
