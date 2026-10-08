"""
Bước 2 — Prompt Hub & A/B Routing
===================================
NHIỆM VỤ:
  1. Viết 2 system prompt khác nhau (V1: ngắn gọn, V2: có cấu trúc)
  2. Push cả 2 lên LangSmith Prompt Hub qua client.push_prompt()
  3. Pull lại từ Hub qua client.pull_prompt()
  4. Implement A/B routing tất định: hash(request_id) % 2 → V1 hoặc V2
  5. Chạy 50 câu hỏi qua router → ≥ 50 LangSmith traces nữa

DELIVERABLE: 2 prompt version hiển thị trong Prompt Hub trên https://smith.langchain.com
"""
import sys
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config  # ⚠️ phải import trước LangChain

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith import Client, traceable

from utils.llm_factory import get_llm, get_embeddings
from utils.data_loader import load_knowledge_base, split_text, build_vectorstore
from qa_pairs import SAMPLE_QUESTIONS


# ── 1. Tên Prompt trên Hub ─────────────────────────────────────────────────
PROMPT_V1_NAME = "nguyenxuantruong-2a202602761-rag-prompt-v1"
PROMPT_V2_NAME = "nguyenxuantruong-2a202602761-rag-prompt-v2"


# ── 2. Định nghĩa 2 Prompt Templates ──────────────────────────────────────
SYSTEM_V1 = (
    "Bạn là trợ lý AI thân thiện. Trả lời ngắn gọn (2-4 câu), chỉ dựa trên context. "
    "Nếu không có thông tin, hãy nói thẳng là không biết.\n\n"
    "Context:\n{context}"
)

PROMPT_V1 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V1),
    ("human",  "{question}"),
])

SYSTEM_V2 = (
    "Bạn là chuyên gia phân tích thông tin. Đọc kỹ context, xác định các facts liên quan, "
    "rồi viết câu trả lời rõ ràng, có tổ chức (3-5 câu). Không suy đoán ngoài context.\n\n"
    "Context:\n{context}"
)

PROMPT_V2 = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_V2),
    ("human",  "{question}"),
])


# ── 3. Push Prompts lên Prompt Hub ─────────────────────────────────────────
def push_prompts_to_hub(client: Client):
    """
    Upload cả 2 prompt templates lên LangSmith Prompt Hub.
    """
    try:
        url = client.push_prompt(PROMPT_V1_NAME, object=PROMPT_V1, description="V1 – ngắn gọn (2-4 câu)")
        print(f"✅ Đã push V1 → {url}")
    except Exception as e:
        if "Nothing to commit" in str(e):
            print(f"ℹ️  V1 '{PROMPT_V1_NAME}' đã có nội dung giống hệt trên Hub")
        else:
            raise RuntimeError(f"Không push được V1 lên Prompt Hub: {e}") from e

    try:
        url = client.push_prompt(PROMPT_V2_NAME, object=PROMPT_V2, description="V2 – có cấu trúc (3-5 câu, expert)")
        print(f"✅ Đã push V2 → {url}")
    except Exception as e:
        if "Nothing to commit" in str(e):
            print(f"ℹ️  V2 '{PROMPT_V2_NAME}' đã có nội dung giống hệt trên Hub")
        else:
            raise RuntimeError(f"Không push được V2 lên Prompt Hub: {e}") from e


# ── 4. Pull Prompts từ Prompt Hub ──────────────────────────────────────────
def pull_prompts_from_hub(client: Client) -> dict:
    """
    Tải 2 prompt từ LangSmith Prompt Hub. Không fallback local vì sẽ
    làm cho lần chạy không còn kiểm chứng được Prompt Hub.
    Trả về: {name: ChatPromptTemplate}
    """
    prompts = {}

    for name in (PROMPT_V1_NAME, PROMPT_V2_NAME):
        try:
            prompts[name] = client.pull_prompt(name)
        except Exception as e:
            raise RuntimeError(f"Không pull được '{name}' từ Prompt Hub: {e}") from e
        print(f"↓ Đã pull '{name}' từ Hub")

    return prompts


# ── 5. A/B Routing tất định ────────────────────────────────────────────────
def get_prompt_version(request_id: str) -> str:
    """
    Xác định prompt version dựa trên MD5 hash của request_id.
    Quy tắc: hash chẵn → PROMPT_V1_NAME | hash lẻ → PROMPT_V2_NAME
    TÍNH CHẤT: cùng request_id LUÔN cho cùng kết quả (deterministic).
    """
    hash_int = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
    return PROMPT_V1_NAME if hash_int % 2 == 0 else PROMPT_V2_NAME


# ── 6. Traced A/B Query ────────────────────────────────────────────────────
@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    """
    Chạy RAG chain với prompt version được chọn bởi router.
    """
    docs = retriever.invoke(question)
    context = "\n\n".join(d.page_content for d in docs)
    answer = (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    return {"question": question, "answer": answer, "version": version}


# ── 7. Setup Vectorstore (tái sử dụng logic Bước 1) ───────────────────────
def setup_vectorstore():
    embeddings  = get_embeddings()
    text        = load_knowledge_base()
    chunk_size  = 600 if config.PROVIDER == "gemini" else 500
    chunks      = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_size // 10)
    return build_vectorstore(chunks, embeddings)


# ── 8. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Bước 2: Prompt Hub & A/B Routing")
    print("=" * 60)

    if not config.validate():
        sys.exit(1)

    client = Client(api_key=config.LANGSMITH_API_KEY)
    push_prompts_to_hub(client)
    prompts = pull_prompts_from_hub(client)

    # Tạo vectorstore, retriever và LLM
    vectorstore = setup_vectorstore()
    retriever   = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm         = get_llm()

    # Chạy A/B routing cho tất cả câu hỏi
    v1_count, v2_count = 0, 0
    for i, question in enumerate(SAMPLE_QUESTIONS):
        request_id  = f"req-{i:04d}"
        version_key = get_prompt_version(request_id)
        version_tag = "v1" if version_key == PROMPT_V1_NAME else "v2"
        prompt      = prompts[version_key]
        result = ask_ab(retriever, llm, prompt, question, version_tag)

        if version_tag == "v1":
            v1_count += 1
        else:
            v2_count += 1
        print(f"[{i+1:02d}] [prompt-{version_tag}] {question[:55]}...")

    print(f"\n📊 Routing: V1={v1_count} câu | V2={v2_count} câu | Tổng={len(SAMPLE_QUESTIONS)}")
    evidence_path = Path(__file__).parent.parent / "evidence" / "02_ab_routing_log.txt"
    evidence_path.write_text(
        "Day 22 — Prompt Hub & A/B Routing\n"
        f"Project: {config.LANGSMITH_PROJECT}\n"
        f"V1: {PROMPT_V1_NAME} | routed={v1_count}\n"
        f"V2: {PROMPT_V2_NAME} | routed={v2_count}\n"
        f"Total: {len(SAMPLE_QUESTIONS)}\n\n"
        "Per-request routing:\n" + "\n".join(
            f"[{i + 1:02d}] request_id=req-{i:04d} prompt-{get_prompt_version(f'req-{i:04d}')[-2:]}"
            for i in range(len(SAMPLE_QUESTIONS))
        ) + "\n",
        encoding="utf-8",
    )
    print(f"📝 Routing evidence đã lưu: {evidence_path}")
    print("✅ Bước 2 hoàn thành! Kiểm tra Prompt Hub và traces trên LangSmith.")


if __name__ == "__main__":
    main()
