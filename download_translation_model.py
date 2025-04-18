import os
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch

def download_translation_model():
    """MBart 모델을 로컬에 다운로드합니다."""
    try:
        # 모델 및 토크나이저 경로 설정
        model_path = r"D:\dataset\fine_tuned_model\translation_model"
        
        # 디렉토리가 없으면 생성
        os.makedirs(model_path, exist_ok=True)
        
        print(f"모델 다운로드 경로: {model_path}")
        print("MBart 모델 다운로드 중...")
        
        # 모델 및 토크나이저 다운로드
        model_name = "facebook/mbart-large-50-many-to-many-mmt"
        
        # 토크나이저 다운로드
        print("토크나이저 다운로드 중...")
        tokenizer = MBart50TokenizerFast.from_pretrained(
            model_name,
            cache_dir=model_path
        )
        
        # 모델 다운로드
        print("모델 다운로드 중...")
        model = MBartForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=model_path
        )
        
        # 모델 및 토크나이저 저장
        print("모델 및 토크나이저 저장 중...")
        tokenizer.save_pretrained(model_path)
        model.save_pretrained(model_path)
        
        print(f"모델 및 토크나이저가 성공적으로 다운로드되었습니다: {model_path}")
        
        # 모델 정보 출력
        print("\n모델 정보:")
        print(f"모델 이름: {model_name}")
        print(f"모델 크기: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M 파라미터")
        print(f"저장 경로: {model_path}")
        
        return True
        
    except Exception as e:
        print(f"모델 다운로드 중 오류 발생: {str(e)}")
        return False

if __name__ == "__main__":
    download_translation_model() 