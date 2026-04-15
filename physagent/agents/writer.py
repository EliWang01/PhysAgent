"""Writer Agent: generates a full academic paper draft from the discovery trajectory."""
import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from physagent.config import API_KEY, BASE_URL, EVALUATOR_MODEL, RETRIEVER_K


WRITER_PROMPT = """你是一位材料学顶尖科学家，同时也是一位经验丰富的学术论文写作者。请基于以下材料发现实验的数据和结果，撰写一篇以发现的材料为主题的学术论文草稿（使用中文）。

【重要】：
- 直接输出论文内容，从标题开始。不要输出任何开场白、寒暄、解释性文字。第一行必须是论文标题。
- 论文的主角是发现的材料及其性质和应用前景，不是 PhysAgent 系统本身。
- PhysAgent 只是研究工具/方法，在方法章节简要介绍即可，不要大篇幅描述系统架构。
- 重点放在材料的科学价值：结构特征、电子性质、热力学稳定性、应用潜力。

【发现目标】：{goal}
【目标属性】：{target_properties}

【最终候选材料】：{best_composition}
【最终计算属性】：{best_properties}
【最终评估结果】：{verdict}

【完整筛选轨迹】：
{trajectory}

【文献参考】：
{context}

【可用参考文献列表】：
{references}

请严格按照以下结构撰写，每个章节都要充分展开，论述详尽：

# 论文标题
（以发现的材料为核心，体现其性质和应用价值。例如："基于计算筛选的CsSn0.5Ge0.5I3卤化物钙钛矿：兼具适宜带隙与热力学稳定性的光伏候选材料"）

## 1. 摘要 (Abstract)
（300-400字。聚焦材料本身：研究背景（该材料体系的重要性）、筛选方法（简述使用了AI辅助的闭环计算筛选）、关键发现（最终材料的具体性质数值）、科学意义和应用前景）

## 2. 引言 (Introduction)
（800-1000字。分3-4段展开：
- 第1段：该材料体系（如卤化物钙钛矿）的研究背景、重要性和广泛应用（光伏、LED等），引用文献数据
- 第2段：当前该材料体系面临的核心科学挑战（如铅毒性、稳定性、带隙调控等），说明为什么需要探索新组分
- 第3段：组分工程和计算筛选在该领域的研究进展，介绍已有的理论和实验工作
- 第4段：本文的研究思路——通过AI辅助的闭环计算筛选方法，系统探索ABX3组分空间，发现了满足目标性质的候选材料，概述论文结构）

## 3. 计算方法 (Computational Methods)
（600-800字。分小节描述：

### 3.1 计算筛选流程
（简要描述使用的AI辅助闭环筛选方法：基于文献的假设生成 → ML势函数结构弛豫与性质计算 → 物理约束验证 → 反馈迭代优化。点到为止，不需要详细描述每个Agent的内部实现）

### 3.2 结构构建与弛豫
（描述晶体结构的构建方法：Pymatgen构建钙钛矿原型结构、混合组分的超胞策略、MACE-MP-0 ML势函数进行结构弛豫的参数设置）

### 3.3 性质计算
（描述各项性质的计算方法：生成能（元素参考态来源）、带隙估算（Goldschmidt容忍因子经验公式及其物理依据）、结构稳定性评估标准）

### 3.4 筛选标准
（列出目标属性的具体阈值和物理约束条件：带隙范围、生成能上限、电荷中性、原子间距、力收敛等））

## 4. 结果与讨论 (Results and Discussion)
（1500-2000字。分小节展开：

### 4.1 组分筛选过程
（逐轮分析筛选过程中探索的候选材料。用表格展示每轮的关键数据（组分、生成能、带隙估算、容忍因子、是否通过）。从材料科学角度解释每次组分调整的物理逻辑——为什么替换某个元素会影响带隙/稳定性）

### 4.2 最终候选材料的结构分析
（深入分析最终材料的晶体结构特征：晶格参数、原子配位环境、Goldschmidt容忍因子、八面体因子。与同族材料对比结构差异）

### 4.3 电子性质与带隙分析
（讨论最终材料的带隙估算值及其物理来源：B位元素的电负性、离子半径对带隙的影响规律。与文献中已知的同类材料带隙数据对比。讨论该带隙值对光伏应用的适宜性（Shockley-Queisser极限））

### 4.4 热力学稳定性分析
（讨论生成能数值的意义：与凸包的距离、分解路径分析。与文献中已知稳定/亚稳态钙钛矿的生成能对比。讨论实际合成的可行性）

### 4.5 应用前景与局限性
（讨论该材料在光伏或其他领域的应用潜力。诚实讨论局限性：带隙估算的精度限制、ML势函数与DFT的差异、需要实验验证的关键性质（如载流子迁移率、缺陷容忍度、环境稳定性）））

## 5. 结论 (Conclusion)
（300-400字。
- 总结通过计算筛选发现的候选材料及其关键性质
- 强调该材料相对于现有材料的优势
- 提出下一步实验验证的建议和优先方向）

## 参考文献
（直接复制下方的参考文献列表到论文中，不要修改、不要添加"待补充"、不要改变格式。原样输出即可。）

{references}
"""


CONTINUATION_PROMPT = """你正在撰写一篇关于 {best_composition} 材料的学术论文。论文的前半部分（到结果与讨论）已经写完。

【重要】：直接输出续写内容，不要输出任何开场白或解释性文字。不要重复已写过的内容。

【论文前文末尾】：
{paper_body}

【材料计算属性】：{best_properties}
【研究目标】：{goal}

【可用参考文献列表】：
{references}

请续写以下章节：

## 5. 结论 (Conclusion)
（300-400字。
- 总结通过计算筛选发现的候选材料及其关键性质数值
- 强调该材料相对于现有材料的优势（如无铅/低铅、适宜带隙、热力学稳定等）
- 提出下一步实验验证的具体建议：建议的合成方法、需要优先验证的性质（载流子迁移率、缺陷容忍度、环境稳定性）、建议的表征手段（XRD、UV-Vis、PL、TRPL等））

## 参考文献
（直接复制下方的参考文献列表到论文中，不要修改、不要添加"待补充"、不要改变格式。原样输出即可。）

{references}
"""


def _format_trajectory(history: list[dict]) -> str:
    """Format iteration history into readable text for the paper."""
    if not history:
        return "无迭代记录"
    lines = []
    for entry in history:
        props = entry.get("computed_properties", {})
        lines.append(
            f"第{entry['iteration']}轮:\n"
            f"  候选材料: {entry['composition']}\n"
            f"  计算属性: formation_energy={props.get('formation_energy', 'N/A')}, "
            f"bandgap_proxy={props.get('bandgap_proxy', 'N/A')}, "
            f"tolerance_factor={props.get('tolerance_factor', 'N/A')}\n"
            f"  评估结果: {entry['verdict']} (score={entry['score']:.3f})\n"
            f"  反馈建议: {entry.get('suggestion', 'N/A')[:150]}"
        )
    return "\n\n".join(lines)


def generate_paper(final_state: dict) -> str:
    """Generate a full academic paper draft from the final state.

    Args:
        final_state: The final AgentState dict from the discovery loop.

    Returns:
        Paper content as markdown string.
    """
    history = final_state.get("history", [])
    goal = final_state.get("goal", "")
    targets = final_state.get("target_properties", {})

    # Find the best result from history
    best_entry = None
    if history:
        valid = [e for e in history if e.get("score", float("inf")) < float("inf")]
        if valid:
            best_entry = min(valid, key=lambda e: e["score"])

    best_composition = final_state.get("composition", "N/A")
    best_properties = final_state.get("computed_properties", {})
    verdict = final_state.get("verdict", "N/A")

    # Clean properties for display (remove numpy arrays)
    clean_props = {}
    for k, v in best_properties.items():
        if k in ("forces", "stress"):
            continue  # Skip large arrays
        clean_props[k] = v

    # Get RAG context with source metadata for references
    context = ""
    references = ""
    try:
        from physagent.memory.rag_store import load_vectorstore
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVER_K})
        docs = retriever.invoke(f"{goal} {best_composition}")

        # Build context with source tags
        context_parts = []
        seen_sources = {}
        ref_idx = 1
        for doc in docs:
            source = doc.metadata.get("source", "未知来源")
            source_name = os.path.basename(source).replace(".pdf", "")
            page = doc.metadata.get("page", "?")
            if source_name not in seen_sources:
                seen_sources[source_name] = ref_idx
                ref_idx += 1
            idx = seen_sources[source_name]
            context_parts.append(f"[文献{idx}, p.{page}] {doc.page_content}")

        context = "\n\n".join(context_parts)

        # Build formatted reference list directly from PDF titles
        ref_lines = []
        for source_name, idx in sorted(seen_sources.items(), key=lambda x: x[1]):
            ref_lines.append(f"[{idx}] {source_name}.")
        references = "\n".join(ref_lines)
    except Exception:
        context = "（文献知识库不可用）"
        references = "（无可用文献）"

    llm = ChatOpenAI(
        api_key=API_KEY, base_url=BASE_URL,
        model=EVALUATOR_MODEL, temperature=0.7,
        max_tokens=8192,
    )

    prompt = ChatPromptTemplate.from_template(WRITER_PROMPT)
    chain = prompt | llm | StrOutputParser()

    invoke_args = {
        "goal": goal,
        "target_properties": json.dumps(targets, ensure_ascii=False),
        "best_composition": best_composition,
        "best_properties": json.dumps(clean_props, ensure_ascii=False, default=str),
        "verdict": verdict,
        "trajectory": _format_trajectory(history),
        "context": context,
        "references": references,
    }

    # First call: main body
    print("    Generating main body...")
    paper_body = chain.invoke(invoke_args)

    # Check if the first call already produced a complete paper (has references section)
    has_references = "## 参考文献" in paper_body or "## References" in paper_body
    has_conclusion = "## 5." in paper_body or "## 结论" in paper_body

    if has_references and has_conclusion:
        # Paper is complete, no need for second call
        print("    Paper generated in one pass.")
        return paper_body

    # Second call: continuation (conclusion + references) only if truncated
    print("    Paper was truncated, generating continuation...")
    continuation_prompt = ChatPromptTemplate.from_template(CONTINUATION_PROMPT)
    continuation_chain = continuation_prompt | llm | StrOutputParser()
    paper_ending = continuation_chain.invoke({
        "paper_body": paper_body[-3000:],
        "references": references,
        "best_composition": best_composition,
        "best_properties": json.dumps(clean_props, ensure_ascii=False, default=str),
        "goal": goal,
    })

    return paper_body + "\n\n" + paper_ending
