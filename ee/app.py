import streamlit as st
import ollama
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import os
import tempfile
import pandas as pd
from typing import List, Dict, Optional
import warnings
import random
import time
warnings.filterwarnings("ignore")

# Streamlit 페이지 설정
st.set_page_config(
    page_title="실전형 업무 시뮬레이터 for 신입",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 스타일 적용
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .role-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .customer-badge {
        background-color: #ff4757;
        color: white;
    }
    .employee-badge {
        background-color: #2ed573;
        color: white;
    }
    .simulation-box {
        border: 2px solid #e1e8ed;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .stats-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ChromaDB 클라이언트 초기화
@st.cache_resource
def init_chroma_client():
    return chromadb.PersistentClient(path="./work_simulator_db")

# Sentence Transformer 모델 초기화
@st.cache_resource
def init_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

# Ollama 연결 확인
def check_ollama_connection():
    try:
        models = ollama.list()
        return True, models
    except Exception as e:
        return False, str(e)

# 문서 처리 함수들
def extract_text_from_pdf(pdf_file) -> str:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            tmp_file_path = tmp_file.name
        
        text = ""
        with open(tmp_file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        os.unlink(tmp_file_path)
        return text
    except Exception as e:
        st.error(f"PDF 텍스트 추출 오류: {str(e)}")
        return ""

def extract_text_from_txt(txt_file) -> str:
    try:
        content = txt_file.getvalue()
        if isinstance(content, bytes):
            content = content.decode('utf-8')
        return content
    except Exception as e:
        st.error(f"TXT 파일 읽기 오류: {str(e)}")
        return ""

def extract_text_from_excel(excel_file) -> str:
    try:
        df = pd.read_excel(excel_file)
        # DataFrame을 문자열로 변환
        text = df.to_string(index=False)
        return text
    except Exception as e:
        st.error(f"Excel 파일 읽기 오류: {str(e)}")
        return ""

def process_uploaded_file(uploaded_file) -> str:
    """업로드된 파일에서 텍스트 추출"""
    file_type = uploaded_file.type
    
    if file_type == "application/pdf":
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "text/plain":
        return extract_text_from_txt(uploaded_file)
    elif file_type in ["application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        return extract_text_from_excel(uploaded_file)
    else:
        st.error(f"지원하지 않는 파일 형식: {file_type}")
        return ""

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():  # 빈 청크 제외
            chunks.append(chunk)
        start = end - overlap
    return chunks

def create_knowledge_base(chunks: List[str], embedding_model, collection_name: str = "work_manual"):
    client = init_chroma_client()
    
    try:
        client.delete_collection(name=collection_name)
    except:
        pass
    
    collection = client.create_collection(name=collection_name)
    
    for i, chunk in enumerate(chunks):
        embedding = embedding_model.encode(chunk).tolist()
        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            ids=[f"chunk_{i}"]
        )
    
    return collection

def search_knowledge_base(query: str, collection, embedding_model, top_k: int = 3) -> List[str]:
    query_embedding = embedding_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0] if results['documents'] else []

# 시뮬레이션 AI 함수들
def generate_customer_scenario(context: str, model_name: str) -> Dict[str, str]:
    """고객 시나리오 생성 (파싱 강화 버전)"""
    try:
        prompt = f"""
당신은 콜센터/매장 고객입니다. 아래 업무 매뉴얼을 참고해서,
실제 업무에서 자주 나올 법한 고객 상황 1가지만 만드세요.

[출력 형식 - 이 형식 그대로, 다른 문장 쓰지 말 것]

상황: (고객이 처한 상황을 한 줄로)
고객 유형: (예: 일반 고객 / 급한 고객 / 까다로운 고객 등)
고객 첫 말: (직원에게 처음 건네는 한 문장)

업무 매뉴얼:
{context[:1500]}
""".strip()

        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        content = response['message']['content'].strip()

        scenario = {
            'situation': '',
            'customer_type': '',
            'first_message': ''
        }

        for line in content.splitlines():
            line = line.strip()
            if line.startswith("상황:"):
                scenario['situation'] = line.split("상황:", 1)[1].strip()
            elif line.startswith("고객 유형:"):
                scenario['customer_type'] = line.split("고객 유형:", 1)[1].strip()
            elif line.startswith("고객 첫 말:") or line.startswith("첫 말:") or "첫 말:" in line:
                scenario['first_message'] = line.split(":", 1)[1].strip().strip('"“”')

        # LLM이 말을 안 듣더라도 기본값 채우기
        if not scenario['situation']:
            scenario['situation'] = "상품과 서비스에 대해 문의하기 위해 전화를 건 고객"
        if not scenario['customer_type']:
            scenario['customer_type'] = "일반 고객"
        if not scenario['first_message']:
            scenario['first_message'] = "안녕하세요, 상품 관련해서 몇 가지 문의드리고 싶습니다."

        return scenario

    except Exception:
        return {
            'situation': '일반적인 문의 상황',
            'customer_type': '일반 고객',
            'first_message': '안녕하세요, 문의사항이 있어서 연락드렸습니다.'
        }


def customer_ai_response(user_message: str, context: str, scenario: Dict, model_name: str) -> str:
    """고객 AI 응답 생성"""
    try:
        prompt = f"""당신은 다음 상황의 고객입니다:

상황: {scenario.get('situation', '')}
고객 유형: {scenario.get('customer_type', '')}

현재까지의 대화와 직원의 응답을 보고, 고객으로서 자연스럽게 대답해주세요.
직원 응답: {user_message}

고객답변 (50자 이내로 간단히):"""

        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content'].strip()
    except Exception as e:
        return "네, 알겠습니다. 감사합니다."

def employee_ai_response(user_message: str, context: str, model_name: str) -> str:
    """직원 AI 응답 생성"""
    try:
        prompt = f"""다음 업무 매뉴얼을 참고하여 고객 문의에 전문적이고 친절하게 응답해주세요:

업무 매뉴얼:
{context}

고객 문의: {user_message}

친절하고 정확한 직원 응답 (100자 이내):"""

        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        return response['message']['content'].strip()
    except Exception as e:
        return "죄송합니다. 확인 후 다시 안내해드리겠습니다."

def evaluate_response(user_response: str, context: str, model_name: str) -> Dict[str, any]:
    """사용자 응답 평가"""
    try:
        prompt = f"""다음 업무 매뉴얼을 기준으로 직원의 고객 응답을 평가해주세요:

업무 매뉴얼:
{context[:1000]}

직원 응답: {user_response}

다음 기준으로 평가해주세요:
1. 정확성 (1-5점)
2. 친절성 (1-5점)  
3. 적절성 (1-5점)
총점: /15점

형식:
정확성: X/5 - 간단한 코멘트
친절성: X/5 - 간단한 코멘트  
적절성: X/5 - 간단한 코멘트
총점: X/15
개선점: 구체적인 개선 제안"""

        response = ollama.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        
        content = response['message']['content']
        
        # 점수 추출
        total_score = 12  # 기본 점수
        try:
            if '총점:' in content:
                score_line = [line for line in content.split('\n') if '총점:' in line][0]
                total_score = int(score_line.split('/')[0].split(':')[-1].strip())
        except:
            pass
        
        return {
            'score': total_score,
            'max_score': 15,
            'feedback': content
        }
    except Exception as e:
        return {
            'score': 10,
            'max_score': 15,
            'feedback': '평가 중 오류가 발생했습니다.'
        }

# 메인 애플리케이션
def main():
    # 헤더
    st.markdown('<div class="main-header">🎯 실전형 업무 시뮬레이터 for 신입</div>', unsafe_allow_html=True)
    st.markdown("### 💼 신입 직원을 위한 고객 응대 연습 도구")
    
    # 사이드바 - 설정 및 문서 업로드
    with st.sidebar:
        st.header("📚 업무 매뉴얼 업로드")
        
        uploaded_files = st.file_uploader(
            "매뉴얼 파일들을 업로드하세요",
            type=['pdf', 'txt', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="PDF, TXT, Excel 파일을 지원합니다"
        )
        
        st.header("⚙️ AI 설정")
        
        # Ollama 연결 확인
        if 'ollama_connected' not in st.session_state:
            with st.spinner("🔍 AI 시스템 연결 확인 중..."):
                connected, result = check_ollama_connection()
                st.session_state['ollama_connected'] = connected
        
        if st.session_state['ollama_connected']:
            st.success("✅ AI 시스템 연결됨")
        else:
            st.error("❌ AI 시스템 연결 실패")
        
        model_name = st.selectbox(
            "AI 모델 선택",
            ["exaone3.5:2.4b-jetson", "llama3.2", "gemma2"],
            index=0
        )
        
        st.header("📊 학습 통계")
        
        # 세션 통계 초기화
        if 'stats' not in st.session_state:
            st.session_state.stats = {
                'total_simulations': 0,
                'customer_role_count': 0,
                'employee_role_count': 0,
                'avg_score': 0,
                'total_score': 0
            }
        
        stats = st.session_state.stats
        
        st.markdown(f"""
        <div class="stats-card">
            <h4>총 시뮬레이션: {stats['total_simulations']}</h4>
        </div>
        <div class="stats-card">
            <h4>고객 역할: {stats['customer_role_count']}</h4>
        </div>
        <div class="stats-card">
            <h4>직원 역할: {stats['employee_role_count']}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # 메인 컨텐츠
    if uploaded_files:
        # 문서 처리
        if st.button("📖 매뉴얼 학습 시작", type="primary"):
            with st.spinner("📚 업무 매뉴얼을 분석하고 있습니다..."):
                all_text = ""
                for file in uploaded_files:
                    text = process_uploaded_file(file)
                    all_text += f"\n\n=== {file.name} ===\n{text}"
                
                if all_text:
                    st.success(f"✅ {len(uploaded_files)}개 파일 처리 완료!")
                    
                    # 지식 베이스 생성
                    embedding_model = init_embedding_model()
                    chunks = chunk_text(all_text)
                    collection = create_knowledge_base(chunks, embedding_model)
                    
                    st.session_state['knowledge_base'] = collection
                    st.session_state['embedding_model'] = embedding_model
                    st.session_state['manual_content'] = all_text
                    
                    st.info(f"📖 총 {len(chunks)}개 학습 단위로 분할 완료")
        
        # 시뮬레이션 섹션
        if 'knowledge_base' in st.session_state:
            st.markdown("---")
            
            # 역할 선택
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="simulation-box">
                    <h3>👤 고객 역할 연습</h3>
                    <p>AI가 직원이 되어 고객인 당신의 문의에 응답합니다.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("고객으로 연습하기", key="customer_practice"):
                    st.session_state.current_role = "customer"
                    st.session_state.simulation_active = True
                    st.session_state.conversation_history = []
                    st.session_state.stats['customer_role_count'] += 1
                    st.rerun()
            
            with col2:
                st.markdown("""
                <div class="simulation-box">
                    <h3>👔 직원 역할 연습</h3>
                    <p>AI가 다양한 고객이 되어 당신이 응대해야 할 상황을 만듭니다.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("직원으로 연습하기", key="employee_practice"):
                    st.session_state.current_role = "employee"
                    st.session_state.simulation_active = True
                    st.session_state.conversation_history = []
                    st.session_state.stats['employee_role_count'] += 1
                    
                    # 고객 시나리오 생성
                    context = search_knowledge_base(
                        "고객 문의", 
                        st.session_state['knowledge_base'],
                        st.session_state['embedding_model']
                    )
                    scenario = generate_customer_scenario(" ".join(context), model_name)
                    st.session_state.customer_scenario = scenario
                    
                    st.rerun()
            
            # 시뮬레이션 실행
            if hasattr(st.session_state, 'simulation_active') and st.session_state.simulation_active:
                st.markdown("---")
                
                # 현재 역할 표시
                if st.session_state.current_role == "customer":
                    st.markdown('<div class="role-badge customer-badge">👤 당신의 역할: 고객</div>', unsafe_allow_html=True)
                    st.markdown("**💡 상황:** AI 직원에게 문의사항을 말해보세요.")
                else:
                    st.markdown('<div class="role-badge employee-badge">👔 당신의 역할: 직원</div>', unsafe_allow_html=True)
                    
                    # 고객 시나리오 표시
                    scenario = st.session_state.get('customer_scenario', {})
                    if scenario:
                        st.markdown(f"""
                        **📋 상황:** {scenario.get('situation', '')}  
                        **👥 고객 유형:** {scenario.get('customer_type', '')}  
                        **💬 고객 첫 말:** "{scenario.get('first_message', '')}"
                        """)
                
                # 대화 히스토리
                if 'conversation_history' not in st.session_state:
                    st.session_state.conversation_history = []
                
                # 직원 모드일 때 첫 고객 메시지 추가
                if (st.session_state.current_role == "employee" and 
                    not st.session_state.conversation_history and 
                    'customer_scenario' in st.session_state):
                    
                    first_msg = st.session_state.customer_scenario.get('first_message', '')
                    if first_msg:
                        st.session_state.conversation_history.append({
                            'role': 'customer_ai',
                            'message': first_msg
                        })
                
                # 대화 표시
                for msg in st.session_state.conversation_history:
                    if msg['role'] == 'user':
                        if st.session_state.current_role == "customer":
                            with st.chat_message("user"):
                                st.markdown(f"**고객 (당신):** {msg['message']}")
                        else:
                            with st.chat_message("user"):
                                st.markdown(f"**직원 (당신):** {msg['message']}")
                    
                    elif msg['role'] == 'employee_ai':
                        with st.chat_message("assistant"):
                            st.markdown(f"**AI 직원:** {msg['message']}")
                    
                    elif msg['role'] == 'customer_ai':
                        with st.chat_message("assistant"):
                            st.markdown(f"**AI 고객:** {msg['message']}")
                
                # 사용자 입력
                if st.session_state.current_role == "customer":
                    user_input = st.chat_input("고객으로서 문의사항을 입력하세요...")
                else:
                    user_input = st.chat_input("직원으로서 응답을 입력하세요...")
                
                if user_input:
                    # 사용자 메시지 추가 (고객/직원 공통)
                    st.session_state.conversation_history.append({
                        'role': 'user',
                        'message': user_input
                    })
                
                    # 매뉴얼 기반 컨텍스트 검색
                    context = search_knowledge_base(
                        user_input,
                        st.session_state['knowledge_base'],
                        st.session_state['embedding_model']
                    )
                    context_text = " ".join(context)
                
                    if st.session_state.current_role == "customer":
                        # 👤 고객 역할: AI가 직원으로 응답
                        ai_response = employee_ai_response(user_input, context_text, model_name)
                        st.session_state.conversation_history.append({
                            'role': 'employee_ai',
                            'message': ai_response
                        })
                
                    else:
                        # 👔 직원 역할: 내가 답변 → 평가 + 다음 고객 질문 자동 생성
                
                        # 1) 내 답변 평가
                        evaluation = evaluate_response(user_input, context_text, model_name)
                        st.session_state.last_evaluation = evaluation
                
                        # 2) 통계 업데이트
                        stats = st.session_state.stats
                        stats['total_score'] += evaluation['score']
                        stats['total_simulations'] += 1
                        stats['avg_score'] = (
                            stats['total_score'] / stats['total_simulations']
                            if stats['total_simulations'] > 0 else 0
                        )
                
                        # 3) 다음 고객 시나리오 생성
                        kb = st.session_state['knowledge_base']
                        emb_model = st.session_state['embedding_model']
                        next_ctx = search_knowledge_base("고객 문의", kb, emb_model)
                        next_scenario = generate_customer_scenario(" ".join(next_ctx), model_name)
                        st.session_state.customer_scenario = next_scenario
                
                        # 4) 새 고객의 "첫 말"을 바로 채팅창에 추가
                        next_first = next_scenario.get('first_message', '')
                        if next_first:
                            st.session_state.conversation_history.append({
                                'role': 'customer_ai',
                                'message': next_first
                            })
                
                    st.rerun()

                
                # 평가 결과 표시 (직원 모드)
                if (st.session_state.current_role == "employee" and 
                    hasattr(st.session_state, 'last_evaluation')):
                    
                    eval_data = st.session_state.last_evaluation
                    
                    st.markdown("### 📊 응답 평가")
                    
                    col_eval1, col_eval2 = st.columns([1, 2])
                    
                    with col_eval1:
                        score_percentage = (eval_data['score'] / eval_data['max_score']) * 100
                        st.metric("점수", f"{eval_data['score']}/{eval_data['max_score']}", f"{score_percentage:.0f}%")
                    
                    with col_eval2:
                        st.text_area("상세 피드백", eval_data['feedback'], height=100, disabled=True)
                    
                    # 통계 업데이트
                    st.session_state.stats['total_score'] += eval_data['score']
                    st.session_state.stats['total_simulations'] += 1
                    if st.session_state.stats['total_simulations'] > 0:
                        st.session_state.stats['avg_score'] = st.session_state.stats['total_score'] / st.session_state.stats['total_simulations']
                
                # 시뮬레이션 종료 버튼
                col_end1, col_end2 = st.columns([1, 1])
                with col_end1:
                    if st.button("🔄 새 시나리오 시작"):
                        st.session_state.conversation_history = []
                        if st.session_state.current_role == "employee":
                            # 새로운 고객 시나리오 생성
                            context = search_knowledge_base(
                                "고객 문의",
                                st.session_state['knowledge_base'],
                                st.session_state['embedding_model']
                            )
                            scenario = generate_customer_scenario(" ".join(context), model_name)
                            st.session_state.customer_scenario = scenario
                        st.rerun()
                
                with col_end2:
                    if st.button("❌ 시뮬레이션 종료"):
                        st.session_state.simulation_active = False
                        st.rerun()
    
    else:
        # 소개 섹션
        st.markdown("""
        ## 🚀 시작하기
        
        ### 1단계: 업무 매뉴얼 업로드
        - 왼쪽 사이드바에서 업무 매뉴얼 파일들을 업로드하세요
        - **지원 형식:** PDF, TXT, Excel 파일
        - **예시:** 고객응대 매뉴얼, FAQ, 서비스 안내서 등
        
        ### 2단계: 역할 선택
        - **👤 고객 역할:** AI 직원과 대화하며 고객 입장 체험
        - **👔 직원 역할:** AI 고객의 문의에 응대하며 실전 연습
        
        ### 3단계: 실전 연습
        - 실제 업무와 유사한 상황에서 연습
        - AI의 실시간 피드백으로 개선점 파악
        - 반복 학습으로 자신감 향상
        
        ---
        
        ## 💡 주요 기능
        
        - ✅ **다양한 문서 지원**: PDF, TXT, Excel 파일 업로드
        - ✅ **양방향 시뮬레이션**: 고객↔직원 역할 전환
        - ✅ **실시간 평가**: AI의 상세한 피드백 제공
        - ✅ **학습 통계**: 연습 진행도 및 성과 추적
        - ✅ **실전 시나리오**: 매뉴얼 기반 다양한 상황 생성
        
        **🎯 지금 바로 매뉴얼을 업로드하고 연습을 시작해보세요!**
        """)

if __name__ == "__main__":
    main()


