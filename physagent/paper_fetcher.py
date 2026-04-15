import os
import re
import sys
import time

# Fix Windows GBK encoding issue with emoji in print statements
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError

SEARCH_TIMEOUT = 60  # 每个来源的搜索超时时间（秒）

# ================= arXiv 搜索与下载 =================

def search_arxiv(query, max_results=10):
    """搜索 arXiv 论文"""
    import arxiv

    print(f"  🔎 正在搜索 arXiv: '{query}'...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    try:
        for paper in client.results(search):
            results.append({
                "title": paper.title,
                "abstract": paper.summary,
                "pdf_url": paper.pdf_url,
                "published": paper.published.strftime("%Y-%m-%d"),
                "source": "arXiv",
                "id": paper.entry_id
            })
        print(f"  ✅ arXiv 找到 {len(results)} 篇论文")
    except Exception as e:
        print(f"  ⚠️ arXiv 搜索出错: {e}")
    return results


# ================= Semantic Scholar 搜索 =================

def search_semantic_scholar(query, max_results=10):
    """搜索 Semantic Scholar 论文"""
    from semanticscholar import SemanticScholar

    print(f"  🔎 正在搜索 Semantic Scholar: '{query}'...")
    results = []
    try:
        sch = SemanticScholar(timeout=SEARCH_TIMEOUT)
        results_raw = sch.search_paper(
            query,
            limit=max_results,
            fields=["title", "abstract", "openAccessPdf", "year", "externalIds"]
        )
        for paper in results_raw:
            if paper.title is None:
                continue
            pdf_url = None
            if paper.openAccessPdf and isinstance(paper.openAccessPdf, dict):
                pdf_url = paper.openAccessPdf.get("url")
            results.append({
                "title": paper.title,
                "abstract": paper.abstract or "",
                "pdf_url": pdf_url,
                "published": str(paper.year) if paper.year else "",
                "source": "Semantic Scholar",
                "id": paper.externalIds.get("DOI", paper.paperId) if paper.externalIds else paper.paperId
            })
        print(f"  ✅ Semantic Scholar 找到 {len(results)} 篇论文")
    except Exception as e:
        print(f"  ⚠️ Semantic Scholar 搜索出错: {e}")
    return results


# ================= PubMed 搜索 =================

def search_pubmed(query, max_results=10):
    """搜索 PubMed 论文"""
    from pymed import PubMed

    print(f"  🔎 正在搜索 PubMed: '{query}'...")
    results = []
    try:
        pubmed = PubMed(tool="AI_Scientist", email="user@example.com")
        results_raw = list(pubmed.query(query, max_results=max_results))
        for article in results_raw:
            pmc_id = getattr(article, 'pmc', None)
            pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/pdf/" if pmc_id else None
            title = str(getattr(article, 'title', ''))
            if not title:
                continue
            results.append({
                "title": title,
                "abstract": str(getattr(article, 'abstract', '') or ''),
                "pdf_url": pdf_url,
                "published": str(getattr(article, 'publication_date', '')),
                "source": "PubMed",
                "id": str(getattr(article, 'pubmed_id', '').split('\n')[0])
            })
        print(f"  ✅ PubMed 找到 {len(results)} 篇论文")
    except Exception as e:
        print(f"  ⚠️ PubMed 搜索出错: {e}")
    return results


# ================= 统一搜索接口 =================

def search_papers(query, max_results=10, sources=None):
    """从多个来源搜索论文，合并去重，每个来源限时 60 秒"""
    if sources is None:
        sources = ["arxiv", "pubmed"]  # Semantic Scholar 在国内访问不稳定，需要时手动加入

    print(f"\n📡 开始从 {len(sources)} 个来源搜索论文（每个来源超时 {SEARCH_TIMEOUT} 秒）...")
    all_results = []

    search_map = {
        "arxiv": ("arXiv", search_arxiv),
        "semantic_scholar": ("Semantic Scholar", search_semantic_scholar),
        "pubmed": ("PubMed", search_pubmed),
    }

    for key in sources:
        if key not in search_map:
            continue
        name, func = search_map[key]
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, query, max_results)
                results = future.result(timeout=SEARCH_TIMEOUT)
                all_results.extend(results)
        except TimeoutError:
            print(f"  ⏰ {name} 搜索超时（{SEARCH_TIMEOUT}s），已跳过")
        except Exception as e:
            print(f"  ⚠️ {name} 搜索失败: {e}")

    # 按标题去重（忽略大小写）
    seen_titles = set()
    unique = []
    for r in all_results:
        normalized = r["title"].lower().strip()
        if normalized not in seen_titles:
            seen_titles.add(normalized)
            unique.append(r)

    downloadable = sum(1 for r in unique if r["pdf_url"])
    print(f"\n📊 搜索完成：共找到 {len(unique)} 篇不重复论文，其中 {downloadable} 篇可下载 PDF")
    return unique


# ================= 文件名清洗 =================

def sanitize_filename(name):
    """清洗文件名，去除非法字符"""
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name[:120].strip()


# ================= PDF 下载 =================

def download_papers(papers, save_dir):
    """下载论文 PDF，返回成功下载的文件路径列表"""
    os.makedirs(save_dir, exist_ok=True)

    downloadable = [p for p in papers if p["pdf_url"]]
    if not downloadable:
        print("❌ 没有可下载的论文（所有论文都在付费墙后面）")
        return []

    print(f"\n⬇️ 开始下载 {len(downloadable)} 篇论文到 {save_dir}/...")
    downloaded = []

    for i, paper in enumerate(downloadable, 1):
        title = sanitize_filename(paper["title"])
        filepath = os.path.join(save_dir, f"{title}.pdf")

        if os.path.exists(filepath):
            print(f"  [{i}/{len(downloadable)}] ⏭️ 已存在: {title[:50]}...")
            downloaded.append(filepath)
            continue

        print(f"  [{i}/{len(downloadable)}] ⬇️ [{paper['source']}] {title[:50]}...")

        try:
            response = requests.get(
                paper["pdf_url"],
                timeout=60,
                headers={"User-Agent": "Mozilla/5.0 (compatible; AI_Scientist/1.0)"}
            )
            if response.status_code == 200 and response.content.startswith(b'%PDF'):
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"      ✅ 成功!")
                downloaded.append(filepath)
            else:
                reason = f"HTTP {response.status_code}" if response.status_code != 200 else "非 PDF 内容"
                print(f"      ❌ 失败 ({reason})")
        except Exception as e:
            print(f"      ⚠️ 下载出错: {e}")

        time.sleep(1)  # 避免请求过快

    print(f"\n🎉 下载完成！成功 {len(downloaded)}/{len(downloadable)} 篇")
    return downloaded


# ================= 独立运行测试 =================

if __name__ == "__main__":
    print("=" * 60)
    print("  📡 论文搜索与下载工具 (arXiv + Semantic Scholar + PubMed)")
    print("=" * 60)

    query = input("\n请输入搜索关键词（英文效果最佳）: ").strip()
    if not query:
        print("未输入关键词，退出。")
        exit()

    max_results = input("每个来源最多搜索几篇 (默认 10): ").strip()
    max_results = int(max_results) if max_results else 10

    papers = search_papers(query, max_results=max_results)

    if not papers:
        print("未找到任何论文。")
        exit()

    print("\n📋 搜索结果：")
    for i, p in enumerate(papers, 1):
        has_pdf = "✅" if p["pdf_url"] else "❌"
        print(f"  {i}. [{p['source']}] {has_pdf} {p['title'][:70]}")

    confirm = input(f"\n是否下载所有可下载的论文？(y/n): ").strip().lower()
    if confirm == 'y':
        save_dir = os.path.join("papers", sanitize_filename(query))
        download_papers(papers, save_dir)
