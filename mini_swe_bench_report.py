import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import boto3
from dotenv import load_dotenv


def _load_env():
    env_file = Path(__file__).resolve().parent / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=True)


def find_eval_result_file(model_dir: Path) -> Path | None:
    """Find the sb-cli report file: swe-bench_{subset}__{split}__{run_id}.json"""
    for f in model_dir.glob("swe-bench_*__*.json"):
        return f
    return None


def parse_eval_results(model_dir: Path) -> dict | None:
    """Find and parse the sb-cli report file in model_dir."""
    report_file = find_eval_result_file(model_dir)
    if report_file is None:
        return None
    try:
        with open(report_file) as f:
            data = json.load(f)
        resolved = data.get("resolved_instances", 0)
        total = data.get("total_instances", 0)
        rate = resolved / total if total else 0.0
        return {"resolved": resolved, "total": total, "resolution_rate": rate}
    except Exception as e:
        print(f"[WARN] Could not parse {report_file}: {e}")
        return None


def _upload_to_s3(local_path: Path, s3_prefix: str):
    try:
        bucket = os.environ.get("AWS_S3_BUCKET_NAME", "")
        s3_key = f"{s3_prefix}/{local_path.name}"
        s3 = boto3.client(
            "s3",
            region_name=os.environ.get("AWS_REGION", ""),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
        )
        s3.upload_file(str(local_path), bucket, s3_key)
        print(f"[UPLOAD] {local_path} → s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"[WARN] Upload failed for {local_path}: {e}")


def generate_report(logs_dir: str, s3_prefix: str | None = None):
    """Walk logs_dir/{provider}/{model}/, collect eval results, write CSV, upload to S3."""
    logs_path = Path(logs_dir)
    rows = []

    for provider_dir in sorted(logs_path.iterdir()):
        if not provider_dir.is_dir():
            continue
        provider = provider_dir.name

        for model_dir in sorted(provider_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name

            eval_results = parse_eval_results(model_dir)

            # Derive subset/split from the sb-cli report filename: swe-bench_{subset}__{split}__*.json
            report_file = find_eval_result_file(model_dir)
            subset, split = "unknown", "unknown"
            if report_file:
                parts = report_file.stem.split("__")  # e.g. ["swe-bench_lite", "dev", "model"]
                if len(parts) >= 2:
                    subset = parts[0].replace("swe-bench_", "")
                    split = parts[1]

            row = {
                "date": datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                "provider": provider,
                "model": model,
                "subset": subset,
                "split": split,
                "resolved": eval_results["resolved"] if eval_results else None,
                "total": eval_results["total"] if eval_results else None,
                "resolution_rate": eval_results["resolution_rate"] if eval_results else None,
            }
            rows.append(row)
            status = (
                f"resolved={row['resolved']}/{row['total']} rate={row['resolution_rate']}"
                if eval_results
                else "no eval_results.json"
            )
            print(f"  {provider}/{model}: {status}")

    if not rows:
        print("[WARN] No results found in", logs_dir)
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    csv_path = logs_path / f"results_{timestamp}.csv"
    fieldnames = ["date", "provider", "model", "subset", "split", "resolved", "total", "resolution_rate"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)

    print(f"[REPORT] CSV written to: {csv_path}")

    if s3_prefix:
        _upload_to_s3(csv_path, s3_prefix)

    return str(csv_path)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mini_swe_bench_report.py <logs_dir> [s3_prefix]")
        sys.exit(1)

    _load_env()
    s3_prefix = sys.argv[2] if len(sys.argv) > 2 else None
    generate_report(sys.argv[1], s3_prefix)
