import os
from dotenv import load_dotenv
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM
)
import torch
import re

# 환경 변수 로드
load_dotenv()

# 모델 저장 경로
SAVE_PATH = r"D:\dataset\fine_tuned_model"

def load_models():
    """모델과 토크나이저를 로드합니다."""
    print("모델 로드 중...")
    
    # QA 모델 로드
    qa_tokenizer = AutoTokenizer.from_pretrained(os.path.join(SAVE_PATH, "qa_model"))
    qa_model = AutoModelForQuestionAnswering.from_pretrained(
        os.path.join(SAVE_PATH, "qa_model"),
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✓ QA 모델 로드 완료")
    
    # 번역 모델 로드
    translation_tokenizer = AutoTokenizer.from_pretrained(os.path.join(SAVE_PATH, "translation_model"))
    translation_model = AutoModelForSeq2SeqLM.from_pretrained(
        os.path.join(SAVE_PATH, "translation_model"),
        torch_dtype=torch.float16,
        device_map="auto"
    )
    print("✓ 번역 모델 로드 완료")
    
    return qa_model, qa_tokenizer, translation_model, translation_tokenizer

def split_sentences(text):
    """텍스트를 문장 단위로 분리합니다."""
    # 문장 구분자 패턴
    pattern = r'(?<=[.!?])\s+|\n+'
    
    # 문장 분리
    sentences = re.split(pattern, text)
    
    # 빈 문장 제거 및 공백 정리
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences

def translate_to_english(text, model, tokenizer):
    """한국어를 영어로 번역합니다."""
    # NLLB 모델용 언어 코드 설정
    tokenizer.src_lang = "kor_Hang"  # 한국어
    
    # 텍스트를 문장으로 분리
    sentences = split_sentences(text)
    translated_sentences = []
    
    for sentence in sentences:
        encoded = tokenizer(sentence, return_tensors="pt")
        # GPU로 입력 이동
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        
        # NLLB 모델의 언어 코드 처리
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
        
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,  # 영어
            max_length=512,
            num_beams=3,  # 메모리 최적화
            early_stopping=True,
            no_repeat_ngram_size=2,  # 메모리 최적화
            length_penalty=1.0,
            temperature=0.7
        )
        
        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        translated_sentences.append(translated)
    
    # 번역된 문장들을 합치기
    return " ".join(translated_sentences)

def translate_to_korean(text, model, tokenizer):
    """영어를 한국어로 번역합니다."""
    # NLLB 모델용 언어 코드 설정
    tokenizer.src_lang = "eng_Latn"  # 영어
    
    # 텍스트를 문장으로 분리
    sentences = split_sentences(text)
    translated_sentences = []
    
    for sentence in sentences:
        encoded = tokenizer(sentence, return_tensors="pt")
        # GPU로 입력 이동
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        
        # NLLB 모델의 언어 코드 처리
        forced_bos_token_id = tokenizer.convert_tokens_to_ids("kor_Hang")
        
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,  # 한국어
            max_length=512,
            num_beams=3,  # 메모리 최적화
            early_stopping=True,
            no_repeat_ngram_size=2,  # 메모리 최적화
            length_penalty=1.0,
            temperature=0.7
        )
        
        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        translated_sentences.append(translated)
    
    # 번역된 문장들을 합치기
    return " ".join(translated_sentences)

def get_answer(question, context, model, tokenizer):
    """질문에 대한 답변을 찾습니다."""
    # 입력 텍스트 토큰화
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="max_length"
    )
    
    # GPU로 입력 이동
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # 모델 추론
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 답변 위치 찾기
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1
    
    # [CLS] 토큰이 선택된 경우 답변 없음으로 처리
    if answer_start == 0 and answer_end == 1:
        return "답변을 찾을 수 없습니다."
    
    # 답변 텍스트 추출
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(
            inputs["input_ids"][0][answer_start:answer_end]
        )
    )
    
    # 빈 답변 처리
    if not answer.strip():
        return "답변을 찾을 수 없습니다."
    
    return answer

def process_qa(question, context, qa_model, qa_tokenizer, translation_model, translation_tokenizer):
    """QA 파이프라인을 처리합니다."""
    # 한국어 질문을 영어로 번역
    translated_question = translate_to_english(question, translation_model, translation_tokenizer)
    translated_context = translate_to_english(context, translation_model, translation_tokenizer)
    
    # 영어로 QA 수행
    answer = get_answer(translated_question, translated_context, qa_model, qa_tokenizer)
    
    # 영어 답변을 한국어로 번역
    if answer != "답변을 찾을 수 없습니다.":
        answer = translate_to_korean(answer, translation_model, translation_tokenizer)
    
    return answer

def main():
    # 모델 로드
    qa_model, qa_tokenizer, translation_model, translation_tokenizer = load_models()
    
    # 테스트
    while True:
        print("\n" + "="*50)
        print("1. 한국어 -> 영어 번역")
        print("2. 영어 -> 한국어 번역")
        print("3. 질문-답변")
        print("4. 종료")
        print("="*50)
        
        choice = input("선택하세요 (1-4): ")
        
        if choice == "1":
            text = input("\n한국어 텍스트를 입력하세요: ")
            translated = translate_to_english(text, translation_model, translation_tokenizer)
            print(f"\n번역 결과: {translated}")
            
        elif choice == "2":
            text = input("\n영어 텍스트를 입력하세요: ")
            translated = translate_to_korean(text, translation_model, translation_tokenizer)
            print(f"\n번역 결과: {translated}")
            
        elif choice == "3":
            context = input("\n문맥을 입력하세요: ")
            question = input("질문을 입력하세요: ")
            answer = process_qa(question, context, qa_model, qa_tokenizer, translation_model, translation_tokenizer)
            print(f"\n답변: {answer}")
            
        elif choice == "4":
            print("\n프로그램을 종료합니다.")
            break
            
        else:
            print("\n잘못된 선택입니다. 다시 선택해주세요.")

if __name__ == "__main__":
    main() 