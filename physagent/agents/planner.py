"""Planner Agent: goal decomposition, RAG-driven hypothesis generation.

Uses DeepSeek-R1 (reasoner) for stronger reasoning capabilities.
"""
import json
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from physagent.config import API_KEY, BASE_URL, PLANNER_MODEL, RETRIEVER_K, AgentState


PLANNER_PROMPT_FIRST = """你是 PhysAgent 的规划者（Planner），一位顶尖的材料科学家。

【任务目标】：{goal}
【目标属性】：{target_properties}

【文献参考】：
{context}

请根据文献知识和材料学原理，提出一个候选材料假设。

【严格限制 - 必须遵守】：
- 仅支持卤化物钙钛矿 ABX3 结构
- A 位只能选：Cs, Rb, K（不支持有机阳离子 MA/FA）
- B 位只能选：Pb, Sn, Ge, Ti, Zr
- X 位只能选：I, Br, Cl, F
- 化学式必须是简单格式，如：CsPbI3, CsSnBr3, CsSn0.5Ge0.5I3, CsPbBr1.5Cl1.5
- 不要使用括号格式如 A(B1B2)X3
- 化学计量必须满足：A=1, B=1, X=3
- 必须引用文献中的依据
- 不允许凭空编造材料性质数据

请严格按以下 JSON 格式输出（不要输出其他内容）：
{{"composition": "化学式如CsPbI3", "space_group": 221, "reasoning": "你的推理过程"}}
"""

PLANNER_PROMPT_REVISE = """你是 PhysAgent 的规划者（Planner），一位顶尖的材料科学家。

【任务目标】：{goal}
【目标属性】：{target_properties}

【上一轮评估反馈】：{suggestion}

【历史尝试记录】：
{history_summary}

【文献参考】：
{context}

请根据评估反馈和文献知识，提出一个新的候选材料假设。

【严格限制 - 必须遵守】：
- 仅支持卤化物钙钛矿 ABX3 结构
- A 位只能选：Cs, Rb, K（不支持有机阳离子 MA/FA）
- B 位只能选：Pb, Sn, Ge, Ti, Zr
- X 位只能选：I, Br, Cl, F
- 化学式必须是简单格式，如：CsPbI3, CsSnBr3, CsSn0.5Ge0.5I3, CsPbBr1.5Cl1.5
- 不要使用括号格式如 A(B1B2)X3
- 化学计量必须满足：A=1, B=1, X=3
- 不要重复已经尝试过的组分
- 必须引用文献中的依据
- 不允许凭空编造材料性质数据
- 如果连续多轮未改善，请尝试不同的元素组合

请严格按以下 JSON 格式输出（不要输出其他内容）：
{{"composition": "化学式", "space_group": 221, "reasoning": "你的推理过程"}}
"""


def _format_history(history: list[dict]) -> str:
    """Format history into a readable summary for the prompt."""
    if not history:
        return "无历史记录"
    lines = []
    for entry in history[-5:]:  # Last 5 iterations
        lines.append(
            f"- 第{entry['iteration']}轮: {entry['composition']} | "
            f"结果: {entry['verdict']} | 分数: {entry['score']:.3f} | "
            f"建议: {entry.get('suggestion', 'N/A')[:80]}"
        )
    return "\n".join(lines)


def _parse_planner_output(text: str) -> dict:
    """Parse JSON output from the Planner LLM.

    Handles cases where the LLM wraps JSON in markdown code blocks.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        return {
            "composition": str(data.get("composition", "")),
            "space_group": int(data.get("space_group", 221)),
            "reasoning": str(data.get("reasoning", "")),
        }
    except (json.JSONDecodeError, ValueError, KeyError):
        return {
            "composition": "CsPbI3",  # Safe default
            "space_group": 221,
            "reasoning": f"Failed to parse LLM output. Raw: {text[:200]}",
        }


def planner_node(state: AgentState) -> dict:
    """LangGraph node function for the Planner Agent."""
    goal = state.get("goal", "")
    targets = state.get("target_properties", {})
    history = state.get("history", [])
    suggestion = state.get("suggestion", "")
    iteration = state.get("iteration", 0)

    llm = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL,
        model=PLANNER_MODEL, temperature=0.7,
    )

    # Try to use RAG if vectorstore is available
    context = ""
    try:
        from physagent.memory.rag_store import load_vectorstore
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
        query = f"{goal} {suggestion}" if suggestion else goal
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
    except Exception:
        context = "（文献知识库不可用，请基于材料学基本原理推理）"

    # Choose prompt based on iteration
    if iteration == 0 or not history:
        prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT_FIRST)
        chain = prompt | llm | StrOutputParser()
        raw_output = chain.invoke({
            "goal": goal,
            "target_properties": json.dumps(targets, ensure_ascii=False),
            "context": context,
        })
    else:
        prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT_REVISE)
        chain = prompt | llm | StrOutputParser()
        raw_output = chain.invoke({
            "goal": goal,
            "target_properties": json.dumps(targets, ensure_ascii=False),
            "suggestion": suggestion,
            "history_summary": _format_history(history),
            "context": context,
        })

    parsed = _parse_planner_output(raw_output)

    print(f"  [Planner] Hypothesis: {parsed['composition']} (SG={parsed['space_group']})")
    print(f"  [Planner] Reasoning: {parsed['reasoning'][:100]}...")

    return {
        "hypothesis": raw_output,
        "composition": parsed["composition"],
        "space_group": parsed["space_group"],
        "reasoning": parsed["reasoning"],
        "simulation_error": None,  # Clear previous error
    }
