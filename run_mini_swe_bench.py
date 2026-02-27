import os
import subprocess
import multiprocessing
import sys
from pathlib import Path
from datetime import datetime

import yaml
from dotenv import load_dotenv

from mini_swe_bench_report import generate_report


CONFIG_FILENAME = "config_swebench.yaml"
BENCH_GROUP_NAME = "mini_swe"

current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
config_path = current_dir / CONFIG_FILENAME
env_file = current_dir / ".env"


def run_export_script():
    """Load .env file from the suite directory into the current process environment."""
    print(f"[SETUP] Loading env from {env_file}")
    if not env_file.exists():
        print(f"[ERROR] {env_file} not found.")
        sys.exit(1)
    load_dotenv(env_file, override=True)
    print("[OK] Environment variables loaded.\n")


def _resolve_env_vars(args: dict) -> dict:
    """Expand values prefixed with $ from the environment."""
    resolved = {}
    for k, v in args.items():
        if isinstance(v, str) and v.startswith("$"):
            v = os.environ.get(v[1:], "")
        resolved[k] = v
    return resolved


def run_command(cmd: list, log_file: Path, dry_run: bool) -> bool:
    """Run a shell command, tee output to log_file, return True on success."""
    cmd_str = " ".join(cmd)
    print(f"[RUNNING] {cmd_str}")
    if dry_run:
        print("[DRY-RUN] Skipping execution.")
        return True
    log_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_file, "w") as lf:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
            )
            lf.write(result.stdout)
            print(result.stdout.strip())
        if result.returncode == 0:
            print(f"[DONE] {cmd_str}")
        else:
            print(f"[FAIL] exit code {result.returncode}: {cmd_str}")
        return result.returncode == 0
    except Exception as e:
        print(f"[EXCEPTION] {cmd_str}: {e}")
        return False


def _upload_file(local_path: Path, s3_prefix: str):
    """Upload a single file to S3 under s3_prefix."""
    try:
        import boto3

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


def run_provider(
    provider: str,
    models: dict,
    swebench_opts: dict,
    base_urls: dict,
    provider_model_args: dict,
    litellm_prefixes: dict,
    overrides: dict,
    output_base: Path,
    s3_prefix: str,
    dry_run: bool,
):
    """Run all models for one provider sequentially."""
    litellm_prefix = litellm_prefixes.get(provider, provider.lower())
    base_url = base_urls.get(provider)

    # Resolve extra per-provider model kwargs (api_key, etc.)
    extra_kwargs = _resolve_env_vars(provider_model_args.get(provider, {}))

    subset = swebench_opts.get("subset", "lite")
    split = swebench_opts.get("split", "dev")
    workers = swebench_opts.get("workers", 4)
    step_limit = swebench_opts.get("step_limit", 100)
    environment_class = swebench_opts.get("environment_class", "local")
    run_eval = swebench_opts.get("run_eval", True)
    slice_spec = swebench_opts.get("slice", "")
    bench_config = swebench_opts.get("bench_config", "swebench.yaml")

    for alias, model_id in models.items():
        if model_id == "not_available":
            print(f"[SKIP] {provider}/{alias}")
            continue

        model_dir = output_base / provider / alias
        model_dir.mkdir(parents=True, exist_ok=True)
        log_file = model_dir / "run.log"
        preds_path = model_dir / "preds.json"
        eval_path = model_dir / "eval_results.json"

        model_str = f"{litellm_prefix}/{model_id}"

        # Per-model step_limit override
        effective_step_limit = (
            overrides.get(provider, {}).get(alias, {}).get("step_limit", step_limit)
        )

        # --- Generation command ---
        # swebench.yaml must be the first -c arg: any -c flag replaces the default config.
        gen_cmd = [
            "mini-extra", "swebench",
            "--model", model_str,
            "--subset", subset,
            "--split", split,
            "--workers", str(workers),
            "-c", bench_config,
            "-c", f"agent.step_limit={effective_step_limit}",
            "-c", f"environment.environment_class={environment_class}",
            "-c", "model.cost_tracking=ignore_errors",
            "-c", 'model.model_kwargs.allowed_openai_params=["tools","parallel_tool_calls"]',
        ]

        if slice_spec:
            gen_cmd += ["--slice", slice_spec]

        if base_url:
            gen_cmd += ["-c", f"model.model_kwargs.base_url={base_url}"]

        for k, v in extra_kwargs.items():
            if v:
                gen_cmd += ["-c", f"model.model_kwargs.{k}={v}"]

        gen_cmd += ["-o", str(model_dir)]

        if dry_run:
            # Mask secret values (api_key and similar) in printed output
            def _mask(cmd):
                masked, skip_next = [], False
                secret_keys = {"api_key", "token", "secret"}
                for token in cmd:
                    if skip_next:
                        masked.append("****")
                        skip_next = False
                    elif token == "-c" :
                        masked.append(token)
                        # peek at next token to check if it's a secret key=value
                    elif token.startswith("model.model_kwargs.") and "=" in token:
                        key = token.split("=", 1)[0].split(".")[-1]
                        if key in secret_keys:
                            masked.append(f"{token.split('=')[0]}=****")
                        else:
                            masked.append(token)
                    else:
                        masked.append(token)
                return masked
            print(f"[DRY-RUN] {provider}/{alias}: {' '.join(_mask(gen_cmd))}")
        else:
            run_command(gen_cmd, log_file, dry_run=False)

        # --- Evaluation command (skipped when run_eval=false to preserve sb-cli quota) ---
        if run_eval:
            eval_cmd = [
                "sb-cli", "submit",
                f"swe-bench_{subset}", split,
                "--predictions_path", str(preds_path),
                "--output_dir", str(model_dir),
            ]

            if dry_run:
                print(f"[DRY-RUN] {provider}/{alias} eval: {' '.join(eval_cmd)}")
            else:
                run_command(eval_cmd, model_dir / "eval.log", dry_run=False)
        else:
            if dry_run:
                print(f"[DRY-RUN] {provider}/{alias} eval: skipped (run_eval=false)")
            else:
                print(f"[SKIP-EVAL] {provider}/{alias} — set run_eval: true to submit")

        # --- Upload artifacts to S3 ---
        if not dry_run:
            model_s3 = f"{s3_prefix}/{provider}/{alias}"
            for artifact in [preds_path, eval_path, log_file]:
                if artifact.exists():
                    _upload_file(artifact, model_s3)


def main():
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        sys.argv = [arg for arg in sys.argv if arg != "--dry-run"]

    run_export_script()

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    swebench_opts = cfg.get("swebench_options", {})
    max_parallel = cfg.get("max_parallel", 1)
    base_urls = cfg.get("base_urls", {})
    provider_model_args = cfg.get("provider_model_args", {})
    litellm_prefixes = cfg.get("litellm_prefixes", {})
    model_mappings = cfg.get("model_mappings", {})
    overrides = cfg.get("overrides", {})

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = current_dir / "logs" / BENCH_GROUP_NAME / run_timestamp
    output_base.mkdir(parents=True, exist_ok=True)

    s3_prefix = f"fc-so-testing-suite/mini_swe_snova/{run_timestamp}"

    print(f"[INFO] Output will be saved locally in: {output_base}")
    print(f"[INFO] Results will be uploaded to: s3://{os.getenv('AWS_S3_BUCKET_NAME')}/{s3_prefix}\n")
    if dry_run:
        print("[INFO] Dry-run enabled: commands will be printed, not executed.\n")

    tasks = [
        (
            provider,
            models,
            swebench_opts,
            base_urls,
            provider_model_args,
            litellm_prefixes,
            overrides,
            output_base,
            s3_prefix,
            dry_run,
        )
        for provider, models in model_mappings.items()
    ]

    if dry_run:
        for task in tasks:
            run_provider(*task)
    else:
        num_processes = min(max_parallel, len(tasks))
        print(f"[INFO] Running {len(tasks)} providers with max {num_processes} parallel processes\n")
        with multiprocessing.Pool(processes=num_processes) as pool:
            pool.starmap(run_provider, tasks)

    if not dry_run:
        generate_report(str(output_base), s3_prefix)

    print(f"\n[COMPLETE] All benchmarks finished.")
    print(f"Local output: {output_base}")
    print(f"S3 prefix:    s3://{os.getenv('AWS_S3_BUCKET_NAME')}/{s3_prefix}")


if __name__ == "__main__":
    main()
