import os
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 환경 변수 로드
load_dotenv()

# 경로 설정
CACHE_DIR = r"D:\dataset\huggingface_cache"

def load_qa_model():
    """QA 모델을 로드합니다."""
    print("QA 모델 로드 중...")
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")
    
    # 모델 로드
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-large",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        cache_dir=CACHE_DIR
    ).to(device)
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        "google/flan-t5-large",
        cache_dir=CACHE_DIR
    )
    
    print("✓ QA 모델 로드 완료")
    
    return model, tokenizer, device

def get_answer(question, model, tokenizer, device, use_context=False):
    """질문에 대한 답변을 생성합니다."""
    try:
        # 입력 텍스트 준비
        if use_context:
            # 컨텍스트를 포함한 입력 (RAG 방식)
            input_text = f"question: {question}\ncontext: I am a software developer with experience in Python, JavaScript, and machine learning. I have worked on various projects including web applications and AI models."
        else:
            # 컨텍스트 없는 입력 (기본 모델 성능 테스트)
            input_text = f"question: {question}"
        
        print(f"\n입력 텍스트: {input_text}")
        
        # 토크나이징
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        # 답변 생성
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=5,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        # 답변 디코딩
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer
        
    except Exception as e:
        print(f"답변 생성 중 오류 발생: {str(e)}")
        return f"오류가 발생했습니다: {str(e)}"

def main():
    """메인 함수"""
    try:
        # QA 모델 로드
        model, tokenizer, device = load_qa_model()
        
        print("\n=== QA 모델 테스트 시작 ===")
        print("종료하려면 'q'를 입력하세요.")
        print("컨텍스트를 사용하려면 'c'를 입력하세요.")
        print("\n예시 질문:")
        print("1. What is your name?")
        print("2. What is your occupation?")
        print("3. What is your educational background?")
        print("4. What is your work experience?")
        print("5. What is your technical stack?")
        print("6. Please describe your project experience.")
        print("7. What are your hobbies?")
        print("8. What are your goals?")
        print("9. What are your strengths?")
        print("10. What are your weaknesses?")
        
        use_context = False
        
        while True:
            # 사용자 입력 받기
            user_input = input("\n질문을 입력하세요 (또는 'c'를 입력하여 컨텍스트 모드 전환): ").strip()
            
            if user_input.lower() == 'q':
                print("프로그램을 종료합니다.")
                break
            
            if user_input.lower() == 'c':
                use_context = not use_context
                print(f"컨텍스트 모드: {'활성화' if use_context else '비활성화'}")
                continue
            
            if not user_input:
                print("질문을 입력해주세요.")
                continue
            
            # 답변 생성
            answer = get_answer(user_input, model, tokenizer, device, use_context)
            print(f"\n답변: {answer}")
            
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 