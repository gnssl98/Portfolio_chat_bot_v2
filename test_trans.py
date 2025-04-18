import os
from dotenv import load_dotenv
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
import PyPDF2
import glob

# 환경 변수 로드
load_dotenv()

def load_translation_model():
    """번역 모델을 로드합니다."""
    print("번역 모델 로드 중...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 중인 디바이스: {device}")
    
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(device)
    print("✓ 번역 모델 로드 완료")
    
    return model, tokenizer, device

def translate_to_english(text, model, tokenizer, device):
    """한국어를 영어로 번역합니다."""
    # 텍스트를 더 작은 청크로 나누기 (약 500자 단위)
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []
    
    for i, chunk in enumerate(chunks, 1):
        print(f"청크 {i}/{len(chunks)} 번역 중...")
        
        # 한국어 설정
        tokenizer.src_lang = "ko_KR"
        
        # 텍스트 토크나이징
        encoded = tokenizer(chunk, return_tensors="pt").to(device)
        
        # 번역 생성
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
            max_length=512,
            num_beams=5,
            early_stopping=True
        )
        
        # 번역 결과 디코딩
        translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        translated_chunks.append(translated)
    
    # 모든 청크를 하나로 합치기
    return " ".join(translated_chunks)

def translate_to_korean(text, model, tokenizer, device):
    """영어를 한국어로 번역합니다."""
    # 텍스트를 더 작은 청크로 나누기 (약 500자 단위)
    chunk_size = 500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []
    
    for i, chunk in enumerate(chunks, 1):
        print(f"청크 {i}/{len(chunks)} 번역 중...")
        
        # 영어 설정
        tokenizer.src_lang = "en_XX"
        
        # 텍스트 토크나이징
        encoded = tokenizer(chunk, return_tensors="pt").to(device)
        
        # 번역 생성
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"]
        )
        
        # 번역 결과 디코딩
        translated_chunk = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        translated_chunks.append(translated_chunk)
    
    # 모든 청크를 합쳐서 반환
    return " ".join(translated_chunks)

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
    pdf_files = glob.glob(os.path.join("./portfolio_data", "**", "*.pdf"), recursive=True)
    print(f"\n발견된 PDF 파일: {pdf_files}")
    for pdf_file in pdf_files:
        texts.append(extract_text_from_pdf(pdf_file))
    
    # txt 파일 처리 (하위 폴더 포함)
    txt_files = glob.glob(os.path.join("./portfolio_data", "**", "*.txt"), recursive=True)
    print(f"\n발견된 TXT 파일: {txt_files}")
    for txt_file in txt_files:
        print(f"\nTXT 파일 처리 중: {txt_file}")
        with open(txt_file, 'r', encoding='utf-8') as file:
            texts.append(file.read())
    
    if not texts:
        print(f"경고: ./portfolio_data에서 PDF나 txt 파일을 찾을 수 없습니다.")
        print(f"현재 디렉토리: {os.getcwd()}")
        print(f"디렉토리 내용: {os.listdir('./portfolio_data')}")
        for root, dirs, files in os.walk('./portfolio_data'):
            print(f"\n{root} 폴더 내용:")
            for file in files:
                print(f"- {file}")
    
    return texts

def main():
    """메인 함수"""
    try:
        # 번역 모델 로드
        model, tokenizer, device = load_translation_model()
        
        # 텍스트 파일 로드
        print("\n텍스트 파일 로드 중...")
        texts = load_text_files()
        
        if not texts:
            print("번역할 텍스트가 없습니다.")
            return
        
        # 각 텍스트 번역
        print("\n=== 번역 시작 ===")
        for i, text in enumerate(texts, 1):
            print(f"\n텍스트 {i}/{len(texts)} 번역 중...")
            print("\n원본 텍스트:")
            print("-" * 50)
            print(text[:500] + "..." if len(text) > 500 else text)
            print("-" * 50)
            
            translated = translate_to_english(text, model, tokenizer, device)
            
            print("\n번역된 텍스트:")
            print("-" * 50)
            print(translated[:500] + "..." if len(translated) > 500 else translated)
            print("-" * 50)
            
            # 번역 결과 저장
            output_file = f"translated_text_{i}.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(translated)
            print(f"\n번역 결과가 {output_file}에 저장되었습니다.")
        
        print("\n=== 번역 완료 ===")
        
    except Exception as e:
        print(f"\n프로그램 실행 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    main() 