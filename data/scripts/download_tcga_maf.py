"""
TCGA-LUAD 数据自动下载脚本
数据来源: GDC Data Portal (open access, 无需登录)
包含：
  1. Masked Somatic Mutation MAF 文件
  2. RNA-seq Gene Expression Quantification (STAR Counts, 含 TPM)
"""

import requests
import json
import os
import shutil

import pandas as pd

# 输出目录
MAF_DIR    = "../input"
RNASEQ_DIR = "../rnaseq"
os.makedirs(MAF_DIR, exist_ok=True)
os.makedirs(RNASEQ_DIR, exist_ok=True)

# 兼容旧变量名
SAVE_DIR = MAF_DIR

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT  = "https://api.gdc.cancer.gov/data"


def query_luad_maf_files():
    """查询 TCGA-LUAD Masked Somatic Mutation MAF 文件列表"""
    filters = {
        "op": "and",
        "content": [
            {
                "op": "=",
                "content": {
                    "field": "cases.project.project_id",
                    "value": "TCGA-LUAD"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Masked Somatic Mutation"
                }
            },
            {
                "op": "=",
                "content": {
                    "field": "data_format",
                    "value": "MAF"
                }
            }
        ]
    }

    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,file_size,cases.submitter_id",
        "format": "JSON",
        "size": "600"
    }

    print("正在查询 GDC 文件列表...")
    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()

    files = response.json()["data"]["hits"]
    print(f"找到 {len(files)} 个 MAF 文件")
    return files


BATCH_SIZE = 50   # GDC bulk endpoint 每批最多提交的文件数

def download_maf_files(files, max_files=None):
    """下载 MAF 文件（分批下载，避免 GDC 500 错误）"""
    if max_files:
        files = files[:max_files]
        print(f"仅下载前 {max_files} 个文件（测试模式）")

    # 分批
    batches = [files[i:i + BATCH_SIZE] for i in range(0, len(files), BATCH_SIZE)]
    print(f"\n共 {len(files)} 个文件，分 {len(batches)} 批下载（每批 {BATCH_SIZE} 个）...")

    for batch_idx, batch in enumerate(batches, 1):
        file_ids = [f["file_id"] for f in batch]
        print(f"\n  [批次 {batch_idx}/{len(batches)}] 下载 {len(file_ids)} 个文件...")

        try:
            response = requests.post(
                GDC_DATA_ENDPOINT,
                data=json.dumps({"ids": file_ids}),
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=300
            )
            response.raise_for_status()
        except Exception as e:
            print(f"  [错误] 批次 {batch_idx} 下载失败: {e}，跳过")
            continue

        archive_path = os.path.join(SAVE_DIR, f"luad_maf_batch{batch_idx}.tar.gz")
        with open(archive_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
        print(f"  下载完成: {archive_path}")

        print("  正在解压...")
        shutil.unpack_archive(archive_path, SAVE_DIR)
        os.remove(archive_path)
        manifest = os.path.join(SAVE_DIR, "MANIFEST.txt")
        if os.path.exists(manifest):
            os.remove(manifest)

        # 重命名本批文件
        rename_maf_files(batch)

    print(f"\n全部完成，文件保存在: {SAVE_DIR}/")


def rename_maf_files(files):
    """将解压后的 MAF 文件从 UUID 子目录移出，重命名为 {case_id}.maf.gz"""
    # 构建 file_id -> case_submitter_id 映射
    id_to_case = {}
    for f in files:
        case_list = f.get("cases", [])
        if case_list:
            id_to_case[f["file_id"]] = case_list[0]["submitter_id"]

    print("\n正在重命名文件...")
    renamed, skipped = 0, 0
    for file_id, case_id in id_to_case.items():
        subdir = os.path.join(SAVE_DIR, file_id)
        if not os.path.isdir(subdir):
            continue

        # 找到子目录中的 MAF 文件（优先取 .maf.gz，否则取 .maf）
        maf_files = [fn for fn in os.listdir(subdir) if fn.endswith(".maf.gz") or fn.endswith(".maf")]
        if not maf_files:
            print(f"  [跳过] {file_id}：未找到 MAF 文件")
            skipped += 1
            continue

        src = os.path.join(subdir, maf_files[0])
        ext = ".maf.gz" if maf_files[0].endswith(".maf.gz") else ".maf"
        dst = os.path.join(SAVE_DIR, f"{case_id}{ext}")

        # 若目标已存在则加后缀避免覆盖
        if os.path.exists(dst):
            base = dst.replace(ext, "")
            i = 2
            while os.path.exists(f"{base}_{i}{ext}"):
                i += 1
            dst = f"{base}_{i}{ext}"

        shutil.move(src, dst)
        shutil.rmtree(subdir)
        print(f"  {maf_files[0]}  ->  {os.path.basename(dst)}")
        renamed += 1

    print(f"重命名完成：{renamed} 个成功，{skipped} 个跳过")


# ─────────────────────────────────────────────────────────────
# RNA-seq 下载部分
# ─────────────────────────────────────────────────────────────

def query_luad_rnaseq_files():
    """查询 TCGA-LUAD RNA-seq STAR Counts 文件列表（Primary Tumor 样本）"""
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-LUAD"}},
            {"op": "=", "content": {"field": "data_category",            "value": "Transcriptome Profiling"}},
            {"op": "=", "content": {"field": "data_type",                "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "analysis.workflow_type",   "value": "STAR - Counts"}},
            {"op": "=", "content": {"field": "cases.samples.sample_type","value": "Primary Tumor"}}
        ]
    }
    params = {
        "filters": json.dumps(filters),
        "fields":  "file_id,file_name,file_size,cases.submitter_id",
        "format":  "JSON",
        "size":    "600"
    }
    print("正在查询 GDC RNA-seq 文件列表...")
    response = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=30)
    response.raise_for_status()
    files = response.json()["data"]["hits"]
    print(f"找到 {len(files)} 个 RNA-seq 文件")
    return files


def download_rnaseq_files(files, max_files=None):
    """批量下载并解压 RNA-seq STAR Counts 文件"""
    if max_files:
        files = files[:max_files]
        print(f"仅下载前 {max_files} 个文件（测试模式）")

    file_ids = [f["file_id"] for f in files]
    print(f"\n开始下载 {len(file_ids)} 个 RNA-seq 文件...")

    response = requests.post(
        GDC_DATA_ENDPOINT,
        data=json.dumps({"ids": file_ids}),
        headers={"Content-Type": "application/json"},
        stream=True,
        timeout=300
    )
    response.raise_for_status()

    archive_path = os.path.join(RNASEQ_DIR, "luad_rnaseq.tar.gz")
    with open(archive_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
    print(f"下载完成: {archive_path}")

    print("正在解压...")
    shutil.unpack_archive(archive_path, RNASEQ_DIR)
    os.remove(archive_path)
    manifest = os.path.join(RNASEQ_DIR, "MANIFEST.txt")
    if os.path.exists(manifest):
        os.remove(manifest)
    print(f"解压完成，文件保存在: {RNASEQ_DIR}/")

    rename_and_convert_rnaseq(files)


def rename_and_convert_rnaseq(files):
    """
    将 UUID 子目录中的 STAR Counts tsv.gz：
    1. 提取 gene_id + tpm_unstranded 列
    2. 转换为 PCGR 所需格式（TargetID + TPM）
    3. 保存为 {case_id}.tsv.gz
    """
    id_to_case = {}
    for f in files:
        case_list = f.get("cases", [])
        if case_list:
            id_to_case[f["file_id"]] = case_list[0]["submitter_id"]

    print("\n正在重命名并转换格式...")
    converted, skipped = 0, 0

    for file_id, case_id in id_to_case.items():
        subdir = os.path.join(RNASEQ_DIR, file_id)
        if not os.path.isdir(subdir):
            continue

        tsv_files = [fn for fn in os.listdir(subdir) if fn.endswith(".tsv.gz") or fn.endswith(".tsv")]
        if not tsv_files:
            print(f"  [跳过] {file_id}：未找到 TSV 文件")
            skipped += 1
            shutil.rmtree(subdir)
            continue

        src = os.path.join(subdir, tsv_files[0])
        dst = os.path.join(RNASEQ_DIR, f"{case_id}.tsv.gz")

        if os.path.exists(dst):
            print(f"  [已存在] {dst}，跳过")
            shutil.rmtree(subdir)
            skipped += 1
            continue

        try:
            df = pd.read_csv(src, sep="\t", comment="#", compression="infer")
            # 过滤 GDC 汇总行（以 N_ 开头，如 N_unmapped）
            df = df[~df.iloc[:, 0].str.startswith("N_")]
            # 提取 gene_id 和 tpm_unstranded，重命名为 PCGR 所需列名
            pcgr_df = df[["gene_id", "tpm_unstranded"]].copy()
            pcgr_df.columns = ["TargetID", "TPM"]
            # 去除 Ensembl 版本号：ENSG00000000003.15 -> ENSG00000000003
            pcgr_df["TargetID"] = pcgr_df["TargetID"].str.replace(r'\.[0-9]+$', '', regex=True)
            pcgr_df.to_csv(dst, sep="\t", index=False, compression="gzip")
            print(f"  {tsv_files[0]}  ->  {case_id}.tsv.gz  ({len(pcgr_df)} genes)")
            converted += 1
        except Exception as e:
            print(f"  [错误] {case_id}：{e}")
            skipped += 1

        shutil.rmtree(subdir)

    print(f"转换完成：{converted} 个成功，{skipped} 个跳过")
    print(f"文件保存在: {RNASEQ_DIR}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--maf-only",  action="store_true", help="只下载 MAF，跳过 RNA-seq")
    parser.add_argument("--rna-only",  action="store_true", help="只下载 RNA-seq，跳过 MAF")
    parser.add_argument("--test",      action="store_true", help="测试模式：每类只下载前3个")
    args = parser.parse_args()

    n = 3 if args.test else None

    if not args.rna_only:
        # ── MAF ──────────────────────────────────────────────
        maf_files = query_luad_maf_files()
        download_maf_files(maf_files, max_files=n)

    if not args.maf_only:
        # ── RNA-seq ───────────────────────────────────────────
        rna_files = query_luad_rnaseq_files()
        download_rnaseq_files(rna_files, max_files=n)

    print("\n完成！")
