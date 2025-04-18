'''
portfolio_data/
├── projects/     # 프로젝트 관련 정보
├── experiences/  # 경력 및 경험
├── skills/       # 기술 스택 및 스킬
└── education/    # 교육 이력
'''

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    MBartForConditionalGeneration,
    MBart50TokenizerFast
)
import torch
import streamlit as st
import re

# 환경 변수 로드
load_dotenv()

# 모델 저장 경로
SAVE_PATH = r"D:\dataset\fine_tuned_model"

# 전역 번역 모델 변수
translation_model = None
translation_tokenizer = None

def load_translation_model():
    """번역 모델을 로드합니다."""
    global translation_model, translation_tokenizer
    
    if translation_model is None or translation_tokenizer is None:
        translation_tokenizer = MBart50TokenizerFast.from_pretrained(os.path.join(SAVE_PATH, "translation_model"))
        translation_model = MBartForConditionalGeneration.from_pretrained(os.path.join(SAVE_PATH, "translation_model"))
    
    return translation_model, translation_tokenizer

def load_qa_model():
    """QA 모델을 로드합니다."""
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(SAVE_PATH, "qa_model"))
    model = AutoModelForQuestionAnswering.from_pretrained(
        os.path.join(SAVE_PATH, "qa_model"),
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def translate_to_english(text):
    """한국어를 영어로 번역합니다."""
    model, tokenizer = load_translation_model()
    tokenizer.src_lang = "ko_KR"
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"]
    )
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated

def translate_to_korean(text):
    """영어를 한국어로 번역합니다."""
    model, tokenizer = load_translation_model()
    tokenizer.src_lang = "en_XX"
    encoded = tokenizer(text, return_tensors="pt")
    generated_tokens = model.generate(
        **encoded,
        forced_bos_token_id=tokenizer.lang_code_to_id["ko_KR"]
    )
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated

def load_and_prepare_data():
    """포트폴리오 데이터를 로드하고 학습 데이터를 준비합니다."""
    print("데이터 로드 중...")
    
    documents = []
    
    # PDF 파일 로드
    try:
        pdf_loader = DirectoryLoader("portfolio_data", glob="**/*.pdf", loader_cls=PyPDFLoader)
        pdf_docs = pdf_loader.load()
        documents.extend(pdf_docs)
        print(f"PDF 파일 {len(pdf_docs)}개 로드 완료")
    except Exception as e:
        print(f"PDF 파일 로딩 중 오류 발생: {str(e)}")
    
    # 텍스트 파일 로드
    try:
        txt_loader = DirectoryLoader(
            "portfolio_data", 
            glob="**/*.txt", 
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        txt_docs = txt_loader.load()
        documents.extend(txt_docs)
        print(f"텍스트 파일 {len(txt_docs)}개 로드 완료")
    except Exception as e:
        print(f"텍스트 파일 로딩 중 오류 발생: {str(e)}")
    
    if not documents:
        raise ValueError("로드된 문서가 없습니다. portfolio_data 폴더에 PDF 또는 텍스트 파일을 추가하세요.")
    
    # 문서 쪼개기
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    print(f"총 {len(texts)}개의 텍스트 청크 생성")
    
    # 학습 데이터 준비
    data = []
    for doc in texts:
        try:
            # 한국어 텍스트를 영어로 번역
            translated_text = translate_to_english(doc.page_content)
            
            # QA 형식의 데이터 생성
            data.append({
                "context": translated_text,
                "question": "Please introduce yourself.",
                "answers": {
                    "text": translated_text,
                    "answer_start": 0
                }
            })
        except Exception as e:
            print(f"문서 처리 중 오류 발생: {str(e)}")
            continue
    
    if not data:
        raise ValueError("처리된 데이터가 없습니다. 문서 내용을 확인하세요.")
    
    return data

def prepare_dataset(dataset, tokenizer):
    """데이터셋을 모델 입력 형식으로 변환합니다."""
    def prepare_train_features(examples):
        # 질문과 문맥을 토큰화
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
        
        # 오프셋 매핑을 사용하여 정답 위치 찾기
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        
        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            # 정답 위치 설정
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            # 문맥의 시작과 끝 인덱스 찾기
            context_start = 0
            while sequence_ids[context_start] != 1:
                context_start += 1
            context_end = len(sequence_ids) - 1
            while sequence_ids[context_end] != 1:
                context_end -= 1
            
            # 정답 위치 설정
            answer_start = examples["answers"]["answer_start"]
            answer_text = examples["answers"]["text"]
            
            # 정답의 시작과 끝 위치 찾기
            start_char = answer_start
            end_char = answer_start + len(answer_text)
            
            # 오프셋 매핑을 사용하여 토큰 위치 찾기
            token_start_index = 0
            while offsets[token_start_index][0] <= start_char:
                token_start_index += 1
            token_start_index -= 1
            
            token_end_index = len(offsets) - 1
            while offsets[token_end_index][1] >= end_char:
                token_end_index -= 1
            token_end_index += 1
            
            tokenized_examples["start_positions"].append(token_start_index)
            tokenized_examples["end_positions"].append(token_end_index)
        
        return tokenized_examples
    
    # 데이터셋 변환
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        remove_columns=dataset.column_names,
    )
    
    return tokenized_dataset

def get_answer(question, context):
    """질문에 대한 답변을 생성합니다."""
    model, tokenizer = load_qa_model()
    
    # 한국어 질문을 영어로 번역
    translated_question = translate_to_english(question)
    
    # 입력 데이터 준비
    inputs = tokenizer(
        translated_question,
        context,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    
    # 답변 생성
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(
                inputs["input_ids"][0][answer_start:answer_end]
            )
        )
    
    # 영어 답변을 한국어로 번역
    translated_answer = translate_to_korean(answer)
    return translated_answer

def main():
    """메인 함수"""
    st.title("포트폴리오 챗봇")
    
    # 세션 상태 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 예시 질문
    example_questions = [
        "자기소개를 해주세요",
        "어떤 프로젝트를 진행했나요?",
        "사용할 수 있는 기술 스택은 무엇인가요?",
        "경력이 어떻게 되나요?",
        "어떤 분야에 관심이 있나요?"
    ]
    
    # 예시 질문 버튼
    st.write("예시 질문:")
    cols = st.columns(5)
    for i, question in enumerate(example_questions):
        if cols[i % 5].button(question):
            st.session_state.messages.append({"role": "user", "content": question})
            answer = get_answer(question, "Please introduce yourself.")
            st.session_state.messages.append({"role": "assistant", "content": answer})
    
    # 채팅 기록 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # 사용자 입력
    if prompt := st.chat_input("질문을 입력하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            answer = get_answer(prompt, "Please introduce yourself.")
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
