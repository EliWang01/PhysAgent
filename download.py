import os
import re
import time
from curl_cffi import requests

PAPER_DATABASE = {
    "全固态电池": [
        {
            "title": "Advancements and challenges in solid-state battery technology: An in-depth review of solid electrolytes and anode innovations",
            "url": "https://www.mdpi.com/2313-0105/10/1/29/pdf"},
        {"title": "Technological advances and market developments of solid-state batteries: a review",
         "url": "https://www.mdpi.com/1996-1944/17/1/239/pdf"},
        {"title": "Development of all-solid-state Li-ion batteries: From key technical areas to commercial use",
         "url": "https://www.mdpi.com/2313-0105/9/3/157/pdf"},
        {"title": "Building better batteries in the solid state: a review",
         "url": "https://www.mdpi.com/1996-1944/12/23/3892/pdf"},
        {"title": "Review of the developments and difficulties in inorganic solid-state electrolytes",
         "url": "https://www.mdpi.com/1996-1944/16/6/2510/pdf"},
        {"title": "Sulfide and oxide inorganic solid electrolytes for all-solid-state Li batteries: a review",
         "url": "https://www.mdpi.com/2079-4991/10/8/1606/pdf"},
        {
            "title": "A Comprehensive Review of Sulfide Solid-State Electrolytes: Properties, Synthesis, Applications, and Challenges",
            "url": "https://www.mdpi.com/2073-4352/15/6/492/pdf"},
        {
            "title": "The next frontier in energy storage: a game-changing guide to advances in solid-state battery cathodes",
            "url": "https://www.mdpi.com/2313-0105/10/1/13/pdf"},
        {
            "title": "An industrial perspective and intellectual property landscape on solid-state battery technology with a focus on solid-state electrolyte chemistries",
            "url": "https://www.mdpi.com/2313-0105/10/1/24/pdf"},
        {"title": "A comparative review of models for all-solid-state li-ion batteries",
         "url": "https://www.mdpi.com/2313-0105/10/5/150/pdf"}
    ],
    "钙钛矿太阳能电池稳定性": [
        {"title": "The recent advancement of outdoor performance of perovskite photovoltaic cells technology",
         "url": "https://www.cell.com/heliyon/pdf/S2405-8440(24)12741-7.pdf"},
        {
            "title": "Thermal Degradation Mechanisms and Stability Enhancement Strategies in Perovskite Solar Cells: A Review",
            "url": "https://arxiv.org/pdf/2509.13700.pdf"},
        {"title": "Review on physical impedance models in perovskite solar cells",
         "url": "https://arxiv.org/pdf/2405.10855.pdf"},
        {
            "title": "A Review on Improving PSC Performance through Charge Carrier Management: Where We Stand and What's Next",
            "url": "https://arxiv.org/pdf/2506.21645.pdf"},
        {
            "title": "Compositional and Interface Engineering of Hybrid Metal Halide Perovskite Thin Films for Solar Cells",
            "url": "https://arxiv.org/pdf/2411.17919.pdf"},
        {
            "title": "Machine learning for accelerating the discovery of high performance low-cost solar cells: a systematic review",
            "url": "https://arxiv.org/pdf/2212.13893.pdf"},
        {"title": "Towards a single-junction non-concentrator metal halide perovskite hot carrier solar cell: review",
         "url": "https://arxiv.org/pdf/2601.16310.pdf"},
        {
            "title": "Beyond Point-like Defects in Bulk Semiconductors: Junction Spectroscopy Techniques for PSCs and 2D Materials",
            "url": "https://arxiv.org/pdf/2602.20764.pdf"},
        {"title": "An autonomous living database for perovskite photovoltaics",
         "url": "https://arxiv.org/pdf/2601.17807.pdf"},
        {"title": "Predicting organic-inorganic halide perovskite photovoltaic performance through machine learning",
         "url": "https://arxiv.org/pdf/2412.09638.pdf"}
    ],
    "MOF用于CO2捕集与转化": [
        {"title": "Progress in adsorption-based CO2 capture by metal–organic frameworks",
         "url": "https://pubs.rsc.org/en/content/articlepdf/2012/cs/c1cs15221a"},
        {"title": "Understanding the opportunities of MOFs for CO2 capture and gas-phase CO2 conversion",
         "url": "https://pubs.rsc.org/en/content/articlepdf/2021/re/d1re00034a"},
        {"title": "Research progress in MOFs in CO2 capture from post-combustion coal-fired flue gas",
         "url": "https://pubs.rsc.org/en/content/articlepdf/2010/mr/d1ta07856a"},
        {"title": "Water-stable MOFs: rational construction and carbon dioxide capture",
         "url": "https://pubs.rsc.org/en/content/articlepdf/2000/ym/d3sc06076d"},
        {"title": "Advanced metal–organic frameworks for superior carbon capture – a critical review",
         "url": "https://pubs.rsc.org/en/content/articlepdf/2024/ta/d4ta03877k"},
        {"title": "A critical review on recent advancements in MOFs for CO2 capture, storage and utilization",
         "url": "https://pubs.rsc.org/en/content/articlepdf/2010/w8/d5ta02338f"},
        {"title": "Advances in amine-functionalized MOFs for carbon capture",
         "url": "https://pubs.rsc.org/en/content/articlepdf/2000/qm/d5ta04991a"},
        {"title": "Amine-functionalized MOFs: structure, synthesis and applications",
         "url": "https://pubs.rsc.org/en/content/articlepdf/2021/xx/c6ra01536k"},
        {"title": "Methodologies for evaluation of MOFs in separation applications",
         "url": "https://pubs.rsc.org/en/content/articlepdf/2015/ra/c5ra07830j"},
        {"title": "Historical and contemporary perspectives on MOFs for gas sensing applications: a review",
         "url": "https://pubs.rsc.org/en/content/articlepdf/2023/su/d2su00152g"}
    ],
    "高熵合金与增材制造": [
        {"title": "Additive manufacturing of high-entropy alloys: a review",
         "url": "https://www.mdpi.com/1099-4300/20/12/937/pdf"},
        {"title": "High entropy alloys manufactured by additive manufacturing",
         "url": "https://www.mdpi.com/2075-4701/10/5/639/pdf"},
        {"title": "Additive manufacturing technologies of high entropy alloys (HEA): Review and prospects",
         "url": "https://www.mdpi.com/1996-1944/16/6/2454/pdf"},
        {"title": "Progress in additive manufacturing of high-entropy alloys",
         "url": "https://www.mdpi.com/1996-1944/17/23/5917/pdf"},
        {
            "title": "Review on the tensile properties and strengthening mechanisms of additive manufactured CoCrFeNi-based high-entropy alloys",
            "url": "https://www.mdpi.com/2075-4701/14/4/437/pdf"},
        {
            "title": "Research advances in additively manufactured high-entropy alloys: microstructure, mechanical properties, and corrosion resistance",
            "url": "https://www.mdpi.com/2075-4701/15/2/136/pdf"},
        {"title": "Advances in nickel-containing high-entropy alloys: from fundamentals to additive manufacturing",
         "url": "https://www.mdpi.com/1996-1944/17/15/3826/pdf"},
        {
            "title": "Recent advances in additive manufacturing of high entropy alloys and their nuclear and wear-resistant applications",
            "url": "https://www.mdpi.com/2075-4701/11/12/1980/pdf"},
        {
            "title": "Bibliometric mapping of literature on high-entropy multicomponent alloys and systematic review of emerging applications",
            "url": "https://www.mdpi.com/1099-4300/24/3/329/pdf"},
        {
            "title": "Harnessing metastability for grain size control in multiprincipal element alloys during additive manufacturing",
            "url": "https://arxiv.org/pdf/2405.03670.pdf"}
    ],
    "二维材料用于储能": [
        {"title": "MXenes from MAX phases: synthesis, hybridization, and advances in supercapacitor applications",
         "url": "https://pubs.rsc.org/en/content/articlepdf/2025/ra/d5ra00271k"},
        {"title": "Graphene-MXene van der Waals heterostructures for high performance supercapacitors",
         "url": "https://www.sciopen.com/local/article_pdf/10.26599/NRE.2024.9120148.pdf"},
        {"title": "Challenges and future prospects of the MXene-based materials for energy storage applications",
         "url": "https://www.mdpi.com/2313-0105/9/2/126/pdf"},
        {
            "title": "Recent advances in two-dimensional MXene for supercapacitor applications: progress, challenges, and perspectives",
            "url": "https://www.mdpi.com/2079-4991/13/5/919/pdf"},
        {
            "title": "Recent advances and strategies in MXene-based electrodes for supercapacitors: applications, challenges and future prospects",
            "url": "https://www.mdpi.com/2079-4991/14/1/62/pdf"},
        {"title": "Effect of MXene nanosheet sticking on supercapacitor device performance",
         "url": "https://www.mdpi.com/2076-3417/14/6/2452/pdf"},
        {"title": "Research progress on MXene-based flexible supercapacitors: A review",
         "url": "https://www.mdpi.com/2073-4352/12/8/1099/pdf"},
        {"title": "Current trends in MXene-based nanomaterials for energy storage and conversion system: a mini review",
         "url": "https://www.mdpi.com/2073-4344/10/5/495/pdf"},
        {"title": "MXene-based nanomaterials for multifunctional applications",
         "url": "https://www.mdpi.com/1996-1944/16/3/1138/pdf"},
        {
            "title": "Next-Generation 2D Materials for Energy Conversion and Storage: MXenes for High-Performance Batteries, Supercapacitors, and Electrocatalysis",
            "url": "https://chemrxiv.org/engage/api-gateway/chemrxiv/assets/orp/resource/item/10.26434/chemrxiv-2025-2964v/pdf"}
    ]
}


def sanitize_filename(name):
    name = re.sub(r'[\\/*?:"<>|]', "", name)
    return name[:120].strip()


def download_papers(base_dir="rag_repo"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    total_papers = sum(len(papers) for papers in PAPER_DATABASE.values())
    current_count = 0

    print(f"========== 开始构建AI素材库，共计 {total_papers} 篇论文 ==========")

    for category, papers in PAPER_DATABASE.items():
        cat_dir = os.path.join(base_dir, category)
        if not os.path.exists(cat_dir):
            os.makedirs(cat_dir)

        print(f"\n📁 正在处理领域: 【{category}】")

        for paper in papers:
            current_count += 1
            title = sanitize_filename(paper["title"])
            url = paper["url"]
            filepath = os.path.join(cat_dir, f"{title}.pdf")

            if os.path.exists(filepath):
                print(f"  [{current_count}/{total_papers}] ⏭️ 已存在跳过: {title[:40]}...")
                continue

            print(f"  [{current_count}/{total_papers}] ⬇️ 正在下载: {title[:40]}...")

            try:
                # 【修改点1】去掉了 stream=True，单篇PDF只有几MB，直接读进内存更稳，避免头部丢失
                response = requests.get(url, impersonate="chrome", timeout=60)

                if response.status_code == 200:
                    # 【修改点2】终极必杀技：检查文件的 "Magic Bytes"
                    # 真正的 PDF 文件，前4个字节必定是 b'%PDF'
                    if response.content.startswith(b'%PDF'):
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                        print(f"      ✅ 成功!")
                    else:
                        print(f"      ❌ 失败 (下载到的不是真正的PDF，可能是网页防爬验证或重定向页)")
                else:
                    print(f"      ❌ 失败 (HTTP {response.status_code})")
            except Exception as e:
                print(f"      ⚠️ 报错: {str(e)}")

            time.sleep(3)

    print("\n🎉 全部下载任务执行完毕！")


if __name__ == "__main__":
    download_papers()