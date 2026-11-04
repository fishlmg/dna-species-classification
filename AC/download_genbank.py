#!/usr/bin/env python3
"""按登录号从 NCBI Nucleotide（GenBank / RefSeq）拉取 FASTA。

读取本目录里的登录号清单，经 Biopython 的 Entrez `efetch` 下载，
写入 `AC/data/<accession>.fasta`。`data.py` 只保留每条序列的前 1000 bp
（`max_len`），因此默认请求 `seq_start=1, seq_stop=1000`，不拉取整条染色体。

NCBI 要求脚本自报联系邮箱，并限制请求频率（无 API key 时约每秒 3 次）。
邮箱用环境变量 `NCBI_EMAIL`，可选 `NCBI_API_KEY`。不要把密钥写进仓库。

只解析清单、不访问网络::

    python download_genbank.py --dry-run

试下少量记录::

    export NCBI_EMAIL="you@example.com"
    python download_genbank.py --limit 5
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LISTS = ("AC.txt", "ac1.txt", "ac2.txt", "ac3.txt")
ACCESSION_RE = re.compile(r"[A-Za-z]{1,3}_\d+(?:\.\d+)?")


def resolve_list(name: str) -> Path:
    path = Path(name)
    if path.is_file():
        return path
    beside_script = HERE / name
    if beside_script.is_file():
        return beside_script
    raise SystemExit(f"找不到序列号清单: {name}")


def parse_accessions(paths: list[Path]) -> tuple[list[str], list[str]]:
    """返回去重后的登录号，以及没有版本号的登录号（便于人工核对）。"""
    ordered: list[str] = []
    seen: set[str] = set()
    missing_version: list[str] = []
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        text = text.replace("，", ",").replace("、", ",")
        for token in ACCESSION_RE.findall(text):
            accession = token.upper()
            if accession in seen:
                continue
            seen.add(accession)
            ordered.append(accession)
            if "." not in accession:
                missing_version.append(accession)
    return ordered, missing_version


def configure_entrez() -> float:
    """设置 Entrez 身份，并返回两次请求之间的间隔（秒）。"""
    email = os.environ.get("NCBI_EMAIL", "").strip()
    if not email:
        raise SystemExit("请设置环境变量 NCBI_EMAIL（NCBI 要求提供联系邮箱）。")

    from Bio import Entrez

    Entrez.email = email
    Entrez.tool = "dna-species-classification"
    api_key = os.environ.get("NCBI_API_KEY", "").strip()
    if api_key:
        Entrez.api_key = api_key
        return 0.12
    return 0.34


def fetch_fasta(accession: str, seq_stop: int) -> str:
    from Bio import Entrez

    last_error: Exception | None = None
    for _ in range(2):
        try:
            handle = Entrez.efetch(
                db="nuccore",
                id=accession,
                rettype="fasta",
                retmode="text",
                seq_start=1,
                seq_stop=seq_stop,
            )
            try:
                payload = handle.read()
            finally:
                handle.close()
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8", errors="replace")
            text = payload.strip()
            if not text.startswith(">"):
                preview = text.splitlines()[0][:180] if text else "空响应"
                raise RuntimeError(preview)
            return text + "\n"
        except Exception as exc:  # NCBI 偶发断连；第二次仍失败再记入失败列表
            last_error = exc
            time.sleep(2)
    raise RuntimeError(str(last_error))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lists", nargs="+", default=list(DEFAULT_LISTS), help="登录号清单，默认 AC.txt ac1.txt ac2.txt ac3.txt")
    parser.add_argument("--output", type=Path, default=HERE / "data", help="FASTA 输出目录，默认 AC/data")
    parser.add_argument("--seq-stop", type=int, default=1000, help="每条序列取到第几位，默认 1000，与 data.py 的 max_len 一致")
    parser.add_argument("--limit", type=int, default=0, help="只下载前 N 条；0 表示清单中的全部去重登录号")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已存在的 fasta")
    parser.add_argument("--dry-run", action="store_true", help="只统计登录号，不访问 NCBI")
    args = parser.parse_args(argv)

    if args.seq_stop < 1:
        raise SystemExit("--seq-stop 需要是正整数")
    if args.limit < 0:
        raise SystemExit("--limit 不能为负")

    lists = [resolve_list(name) for name in args.lists]
    accessions, missing_version = parse_accessions(lists)
    if args.limit:
        accessions = accessions[: args.limit]
        selected = set(accessions)
        missing_version = [item for item in missing_version if item in selected]

    print(f"清单 {len(lists)} 个，待处理登录号 {len(accessions)} 个，窗口 1..{args.seq_stop}")
    for path in lists:
        print(f"  - {path}")
    if missing_version:
        print("以下登录号没有版本号（ac3.txt 里有一处全角逗号，版本号被截断），将原样交给 NCBI：")
        for accession in missing_version:
            print(f"  - {accession}")
    if args.dry_run:
        print("dry-run：未访问 NCBI。")
        return 0

    pause = configure_entrez()
    args.output.mkdir(parents=True, exist_ok=True)
    saved = 0
    skipped = 0
    failed: list[tuple[str, str]] = []

    for index, accession in enumerate(accessions, start=1):
        destination = args.output / f"{accession}.fasta"
        if destination.exists() and destination.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            print(f"[{index}/{len(accessions)}] 已存在，跳过 {destination.name}")
            continue
        try:
            fasta = fetch_fasta(accession, args.seq_stop)
        except Exception as exc:
            failed.append((accession, str(exc).splitlines()[0][:200]))
            print(f"[{index}/{len(accessions)}] 失败 {accession}: {failed[-1][1]}", file=sys.stderr)
        else:
            destination.write_text(fasta, encoding="utf-8")
            saved += 1
            print(f"[{index}/{len(accessions)}] 写入 {destination.name}")
        time.sleep(pause)

    print(f"完成：写入 {saved}，跳过 {skipped}，失败 {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
