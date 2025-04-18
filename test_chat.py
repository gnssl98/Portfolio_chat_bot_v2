from chatbot import load_documents, create_vector_store, create_qa_chain
import os
from dotenv import load_dotenv

def test_chat():
    # 환경 변수 로드
    load_dotenv()
    
    print("문서 로드 중...")
    documents = load_documents("portfolio_data")
    print(f"로드된 문서 수: {len(documents)}")
    
    print("\n벡터 저장소 생성 중...")
    vectorstore = create_vector_store(documents)
    print("벡터 저장소 생성 완료")
    
    print("\nQA 체인 생성 중...")
    qa_chain = create_qa_chain(vectorstore)
    print("QA 체인 생성 완료")
    
    print("\n챗봇 테스트를 시작합니다. 종료하려면 'quit'를 입력하세요.")
    
    while True:
        question = input("\n질문을 입력하세요: ")
        if question.lower() == 'quit':
            break
            
        print("\n답변 생성 중...")
        response = qa_chain.run(question)
        print(f"\n답변: {response}")

if __name__ == "__main__":
    test_chat() 