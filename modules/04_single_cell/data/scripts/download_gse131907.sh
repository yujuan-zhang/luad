#!/usr/bin/env bash
# ============================================================
# download_gse131907.sh
# 下载 LUAD 单细胞数据集 GSE131907
# Kim et al., Nature Communications 2020
# 约 50,000 cells, LUAD/LSCC/正常肺组织
#
# 文件说明：
#   raw_UMI_matrix.txt.gz      原始 UMI 计数矩阵（390MB，推荐用于分析）
#   cell_annotation.txt.gz     细胞注释（细胞类型、样本、患者信息）
#   Feature_Summary.xlsx       基因特征说明
#
# 用法：
#   bash download_gse131907.sh
# ============================================================

set -euo pipefail

BASE_URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE131nnn/GSE131907/suppl"
OUT_DIR="$(dirname "$0")/../input/GSE131907"
mkdir -p "${OUT_DIR}"

echo "=== 下载 GSE131907 LUAD 单细胞数据 ==="

# 1. 原始 UMI 矩阵（txt格式，scanpy/Python 可直接读取）
FILE1="GSE131907_Lung_Cancer_raw_UMI_matrix.txt.gz"
if [[ ! -f "${OUT_DIR}/${FILE1}" ]]; then
    echo "[1/3] 下载原始 UMI 矩阵（~390MB）..."
    curl -C - -L -o "${OUT_DIR}/${FILE1}" "${BASE_URL}/${FILE1}"
    echo "[1/3] 完成"
else
    echo "[1/3] 已存在，跳过"
fi

# 2. 细胞注释文件（细胞类型标签，BayesPrism 参考必需）
FILE2="GSE131907_Lung_Cancer_cell_annotation.txt.gz"
if [[ ! -f "${OUT_DIR}/${FILE2}" ]]; then
    echo "[2/3] 下载细胞注释文件（~1.8MB）..."
    curl -C - -L -o "${OUT_DIR}/${FILE2}" "${BASE_URL}/${FILE2}"
    echo "[2/3] 完成"
else
    echo "[2/3] 已存在，跳过"
fi

# 3. 基因特征说明
FILE3="GSE131907_Lung_Cancer_Feature_Summary.xlsx"
if [[ ! -f "${OUT_DIR}/${FILE3}" ]]; then
    echo "[3/3] 下载基因特征说明（~19KB）..."
    curl -C - -L -o "${OUT_DIR}/${FILE3}" "${BASE_URL}/${FILE3}"
    echo "[3/3] 完成"
else
    echo "[3/3] 已存在，跳过"
fi

echo ""
echo "=== 下载完成 ==="
echo "文件位置: ${OUT_DIR}/"
ls -lh "${OUT_DIR}/"
