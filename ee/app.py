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

# 선택적 LLM 라이브러리
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None

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

# -----------------------------
# Chroma / Embedding 초기화
# -----------------------------
@st.cache_resource
def init_chroma_client():
    return chromadb.PersistentClient(path="./work_simulator_db")

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

# -----------------------------
# 문서 처리 함수
# -----------------------------
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
    elif file_type in [
        "application/vnd.ms-excel",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    ]:
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
        if chunk.strip():
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

# -----------------------------
# 공통 LLM 호출 함수
# -----------------------------
def call_llm(prompt: str) -> str:
    provider = st.session_state.get("llm_provider", "ollama")
    model_name = st.session_state.get("model_name", "exaone3.5:2.4b-jetson")

    # 1) 로컬 Ollama
    if provider == "ollama":
        try:
            resp = ollama.chat(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return resp["message"]["content"].strip()
        except Exception as e:
            st.error(f"로컬 Ollama 호출 중 오류: {e}")
            return ""

    # 2) OpenAI GPT
    elif provider == "openai":
        api_key = st.session_state.get("openai_api_key", "")
        if not api_key:
            st.error("OpenAI API Key를 입력하세요.")
            return ""
        if OpenAI is None:
            st.error("openai 패키지가 설치되어 있지 않습니다. `pip install openai` 후 다시 실행하세요.")
            return ""
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"OpenAI 호출 중 오류: {e}")
            return ""

    # 3) Google Gemini
    elif provider == "gemini":
        api_key = st.session_state.get("gemini_api_key", "")
        if not api_key:
            st.error("Gemini API Key를 입력하세요.")
            return ""
        if genai is None:
            st.error("google-generativeai 패키지가 설치되어 있지 않습니다. `pip install google-generativeai` 후 다시 실행하세요.")
            return ""
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            text = getattr(response, "text", "") or ""
            return text.strip()
        except Exception as e:
            st.error(f"Gemini 호출 중 오류: {e}")
            return ""

    else:
        st.error("지원하지 않는 LLM 공급자입니다.")
        return ""

# -----------------------------
# 시뮬레이션 AI 함수들
# -----------------------------
def generate_customer_scenario(context: str) -> Dict[str, str]:
    """업로드한 매뉴얼 내용을 바탕으로 고객 시나리오 생성"""
    try:
        prompt = f"""
당신은 아래 매뉴얼에 나오는 서비스/업무의 고객 또는 사용자입니다.

[업무/서비스 매뉴얼 발췌]
{context[:1500]}

위 매뉴얼의 주제와 용어를 벗어나지 말고,
실제 현장에서 자주 나올 법한 고객 문의 상황 1개만 만드세요.

반드시 매뉴얼의 내용과 직접 관련된 문의여야 하며,
매뉴얼에 없는 새로운 종류의 상품/서비스(옷, 재킷, 음식, 택배, 항공권 등)는 만들지 마세요.

[출력 형식 - 이 형식 그대로]
상황: (고객이 처한 상황을 한 줄로)
고객 유형: (예: 일반 고객 / 초보 학습자 / 컴퓨터에 익숙하지 않은 고객 등)
고객 첫 말: (직원에게 처음 건네는 한 문장)
""".strip()

        content = call_llm(prompt).strip()

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
            elif line.startswith("고객 첫 말:") or line.startswith("첫 말:") or "고객 첫 말:" in line:
                scenario['first_message'] = line.split(":", 1)[1].strip().strip('"“”')

        if not scenario['situation']:
            scenario['situation'] = "매뉴얼에 나온 내용을 문의하기 위해 연락한 고객"
        if not scenario['customer_type']:
            scenario['customer_type'] = "일반 고객"
        if not scenario['first_message']:
            scenario['first_message'] = "안녕하세요, 매뉴얼 내용 관련해서 몇 가지 문의드리고 싶습니다."

        return scenario

    except Exception:
        return {
            'situation': '일반적인 문의 상황',
            'customer_type': '일반 고객',
            'first_message': '안녕하세요, 문의사항이 있어서 연락드렸습니다.'
        }

def customer_ai_response(user_message: str, context: str, scenario: Dict) -> str:
    """고객 AI 응답 생성 (매뉴얼 기반)"""
    try:
        prompt = f"""당신은 다음 상황의 고객입니다.

[업무/서비스 매뉴얼 발췌]
{context[:800]}

상황: {scenario.get('situation', '')}
고객 유형: {scenario.get('customer_type', '')}

위 매뉴얼의 주제와 용어를 벗어나지 말고,
직원의 답변을 들은 뒤 이어질 다음 고객 질문/반응을 한 문장으로만 작성하세요.
매뉴얼에 없는 새로운 종류의 상품/서비스(옷, 재킷, 음식, 택배 등)는 언급하지 마세요.

직원 응답: {user_message}

고객 답변 (50자 이내, 한 문장):"""

        return call_llm(prompt).strip()
    except Exception:
        return "네, 알겠습니다. 안내해 주신 내용으로 진행해 볼게요."

def employee_ai_response(user_message: str, context: str) -> str:
    """직원 AI 응답 생성"""
    try:
        prompt = f"""다음 업무 매뉴얼을 참고하여 고객 문의에 전문적이고 친절하게 응답해주세요:

업무 매뉴얼:
{context}

고객 문의: {user_message}

친절하고 정확한 직원 응답 (100자 이내):"""

        return call_llm(prompt).strip()
    except Exception:
        return "죄송합니다. 확인 후 다시 안내해드리겠습니다."

def evaluate_response(user_response: str, context: str) -> Dict[str, any]:
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

        content = call_llm(prompt)

        total_score = 12
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
    except Exception:
        return {
            'score': 10,
            'max_score': 15,
            'feedback': '평가 중 오류가 발생했습니다.'
        }

# -----------------------------
# 메인 애플리케이션
# -----------------------------
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

        # 임베딩 학습 수준 설정
        st.header("🧠 임베딩 설정")
        embed_percent = st.slider(
            "파일 임베딩 학습 수준 (%)",
            min_value=20,
            max_value=100,
            value=100,
            step=20,
            help="매뉴얼 전체 텍스트 중 임베딩에 사용할 비율입니다."
        )
        st.session_state["embed_ratio"] = embed_percent / 100.0

        # LLM 설정
        st.header("⚙️ AI 설정")

        llm_provider = st.selectbox(
            "LLM 공급자",
            options=["ollama", "openai", "gemini"],
            format_func=lambda v: {
                "ollama": "로컬(Ollama)",
                "openai": "OpenAI GPT",
                "gemini": "Google Gemini"
            }[v]
        )
        st.session_state["llm_provider"] = llm_provider

        model_name = None

        if llm_provider == "ollama":
            if 'ollama_connected' not in st.session_state:
                with st.spinner("🔍 로컬 Ollama 연결 확인 중..."):
                    connected, result = check_ollama_connection()
                    st.session_state['ollama_connected'] = connected

            if st.session_state.get('ollama_connected', False):
                st.success("✅ 로컬 Ollama 연결됨")
            else:
                st.warning("⚠️ Ollama 연결에 실패했습니다. 로컬 모델 사용 시 Ollama 서버를 확인하세요.")

            model_name = st.selectbox(
                "Ollama 모델 선택",
                ["exaone3.5:2.4b-jetson", "llama3.2", "gemma2"],
                index=0
            )

        elif llm_provider == "openai":
            openai_key = st.text_input("OpenAI API Key", type="password")
            st.session_state["openai_api_key"] = openai_key

            openai_models = [
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4o-mini",
                "gpt-4o",
            ]
            model_name = st.selectbox(
                "OpenAI GPT 모델 선택",
                options=openai_models,
                index=0
            )

        elif llm_provider == "gemini":
            gemini_key = st.text_input("Gemini API Key", type="password")
            st.session_state["gemini_api_key"] = gemini_key

            gemini_models = [
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ]
            model_name = st.selectbox(
                "Gemini 모델 선택",
                options=gemini_models,
                index=0
            )

        if model_name:
            st.session_state["model_name"] = model_name

        # 학습 통계
        st.header("📊 학습 통계")
        if 'stats' not in st.session_state:
            st.session_state.stats = {
                'total_simulations': 0,
                'customer_role_count': 0,
                'employee_role_count': 0,
                'avg_score': 0.0,
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

                    embedding_model = init_embedding_model()
                    chunks = chunk_text(all_text)

                    ratio = st.session_state.get("embed_ratio", 1.0)
                    use_n = max(1, int(len(chunks) * ratio))
                    chunks_to_use = chunks[:use_n]

                    collection = create_knowledge_base(chunks_to_use, embedding_model)

                    st.session_state['knowledge_base'] = collection
                    st.session_state['embedding_model'] = embedding_model
                    st.session_state['manual_content'] = all_text

                    st.info(
                        f"📖 총 {len(chunks)}개 청크 중 {use_n}개를 임베딩했습니다. "
                        f"(학습 수준 {int(ratio * 100)}%)"
                    )

        # 시뮬레이션 섹션
        if 'knowledge_base' in st.session_state:
            st.markdown("---")

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

                    manual_text = st.session_state.get('manual_content', '')
                    scenario = generate_customer_scenario(manual_text)
                    st.session_state.customer_scenario = scenario

                    st.rerun()

            # 시뮬레이션 실행
            if hasattr(st.session_state, 'simulation_active') and st.session_state.simulation_active:
                st.markdown("---")

                if st.session_state.current_role == "customer":
                    st.markdown('<div class="role-badge customer-badge">👤 당신의 역할: 고객</div>', unsafe_allow_html=True)
                    st.markdown("**💡 상황:** AI 직원에게 문의사항을 말해보세요.")
                else:
                    st.markdown('<div class="role-badge employee-badge">👔 당신의 역할: 직원</div>', unsafe_allow_html=True)
                    scenario = st.session_state.get('customer_scenario', {})
                    if scenario:
                        st.markdown(f"""
                        **📋 상황:** {scenario.get('situation', '')}  
                        **👥 고객 유형:** {scenario.get('customer_type', '')}  
                        **💬 고객 첫 말:** "{scenario.get('first_message', '')}"
                        """)

                if 'conversation_history' not in st.session_state:
                    st.session_state.conversation_history = []

                # 직원 모드 첫 턴: 고객 첫 발화 자동 추가
                if (st.session_state.current_role == "employee"
                    and not st.session_state.conversation_history
                    and 'customer_scenario' in st.session_state):
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

                # 입력창
                if st.session_state.current_role == "customer":
                    user_input = st.chat_input("고객으로서 문의사항을 입력하세요...")
                else:
                    user_input = st.chat_input("직원으로서 응답을 입력하세요...")

                if user_input:
                    st.session_state.conversation_history.append({
                        'role': 'user',
                        'message': user_input
                    })

                    context = search_knowledge_base(
                        user_input,
                        st.session_state['knowledge_base'],
                        st.session_state['embedding_model']
                    )
                    context_text = " ".join(context)

                    if st.session_state.current_role == "customer":
                        ai_response = employee_ai_response(user_input, context_text)
                        st.session_state.conversation_history.append({
                            'role': 'employee_ai',
                            'message': ai_response
                        })
                    else:
                        # 직원 모드: 답변 → 평가 → 새 질문
                        evaluation = evaluate_response(user_input, context_text)
                        st.session_state.last_evaluation = evaluation

                        stats = st.session_state.stats
                        stats['total_score'] += evaluation['score']
                        stats['total_simulations'] += 1
                        if stats['total_simulations'] > 0:
                            stats['avg_score'] = stats['total_score'] / stats['total_simulations']

                        manual_text = st.session_state.get('manual_content', '')
                        next_scenario = generate_customer_scenario(manual_text)
                        st.session_state.customer_scenario = next_scenario

                        next_first = next_scenario.get('first_message', '')
                        if next_first:
                            st.session_state.conversation_history.append({
                                'role': 'customer_ai',
                                'message': next_first
                            })

                    st.rerun()

                # 평가 결과 (직원 모드만)
                if (st.session_state.current_role == "employee"
                    and hasattr(st.session_state, 'last_evaluation')):
                    eval_data = st.session_state.last_evaluation
                    st.markdown("### 📊 응답 평가")
                    col_eval1, col_eval2 = st.columns([1, 2])
                    with col_eval1:
                        score_percentage = (eval_data['score'] / eval_data['max_score']) * 100
                        st.metric("점수", f"{eval_data['score']}/{eval_data['max_score']}", f"{score_percentage:.0f}%")
                    with col_eval2:
                        st.text_area("상세 피드백", eval_data['feedback'], height=100, disabled=True)

                # 종료 / 새 시나리오
                col_end1, col_end2 = st.columns([1, 1])
                with col_end1:
                    if st.button("🔄 새 시나리오 시작"):
                        st.session_state.conversation_history = []
                        if st.session_state.current_role == "employee":
                            manual_text = st.session_state.get('manual_content', '')
                            scenario = generate_customer_scenario(manual_text)
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
