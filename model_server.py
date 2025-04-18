from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model():
    """모델과 토크나이저를 로드합니다."""
    # 한국어 챗봇에 적합한 모델 선택
    model_name = "beomi/KoAlpaca-Polyglot-1.3B-v2"  # 정확한 모델 이름
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # GPU 사용 가능한 경우 GPU로 이동
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        print(f"모델이 {device}에 로드되었습니다.")
        return model, tokenizer
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {str(e)}")
        raise

def generate_response(model, tokenizer, prompt):
    """프롬프트에 대한 답변을 생성합니다."""
    try:
        # 입력 텍스트 준비
        input_text = f"질문: {prompt}\n답변:"
        
        # 토큰화
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 답변 생성
        outputs = model.generate(
            **inputs,
            max_length=512,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 답변 디코딩
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 입력 프롬프트 부분 제거
        response = response.replace(input_text, "").strip()
        
        return response
    
    except Exception as e:
        print(f"답변 생성 중 오류 발생: {str(e)}")
        return "죄송합니다. 답변을 생성하는 중 오류가 발생했습니다."

def main():
    print("모델을 로드하는 중...")
    model, tokenizer = load_model()
    print("\n챗봇이 준비되었습니다. 종료하려면 'quit'를 입력하세요.")
    
    while True:
        # 사용자 입력 받기
        user_input = input("\n질문을 입력하세요: ")
        
        if user_input.lower() == 'quit':
            print("챗봇을 종료합니다.")
            break
        
        # 답변 생성
        print("\n답변을 생성하는 중...")
        response = generate_response(model, tokenizer, user_input)
        print(f"\n답변: {response}")

if __name__ == "__main__":
    main() 