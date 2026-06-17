from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

from src.tools.evaluator import MODEL_PATHS, validate_model_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "model_delivery.yaml"
RUNTIME_MANIFEST = "_bundle_manifest.json"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_delivery_config(path: str | Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    config_path = resolve_project_path(path)

    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def utc_version() -> str:
    return "v" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


def git_metadata() -> Dict[str, Any]:
    def run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=PROJECT_ROOT,
                check=True,
                text=True,
                capture_output=True,
            )
        except Exception:
            return None

        return result.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    status = run_git(["status", "--short"])

    return {
        "commit": commit,
        "dirty": bool(status),
    }


def iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def file_record(path: Path, root: Path) -> Dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_manifest(
    *,
    model_root: Path,
    bundle_name: str,
    bundle_version: str,
    runtime_model_subdir: str,
) -> Dict[str, Any]:
    readiness = validate_model_artifacts(model_root)

    if readiness["status"] != "ok":
        raise RuntimeError(
            "Cannot bundle incomplete model panel: "
            f"{readiness['missing']}"
        )

    datasets = {}
    files = []

    for short_name, dataset_dir in MODEL_PATHS.items():
        artifact_dir = model_root / dataset_dir
        metadata = read_json(artifact_dir / "metadata.json")
        metrics = read_json(artifact_dir / "metrics.json")

        dataset_files = [
            file_record(path, model_root)
            for path in iter_files(artifact_dir)
        ]
        files.extend(dataset_files)

        datasets[short_name] = {
            "dataset_dir": dataset_dir,
            "task_type": metadata.get("task_type"),
            "primary_metric": metadata.get("primary_metric"),
            "model_type": metadata.get("model_type"),
            "metrics": metrics,
            "files": dataset_files,
        }

    return {
        "schema_version": 1,
        "bundle_name": bundle_name,
        "bundle_version": bundle_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_model_subdir": runtime_model_subdir,
        "required_models": MODEL_PATHS,
        "git": git_metadata(),
        "datasets": datasets,
        "files": files,
    }


def create_bundle(
    *,
    config: Dict[str, Any],
    version: str | None = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    panel_cfg = config.get("model_panel", {})
    bundle_cfg = config.get("bundle", {})

    bundle_name = panel_cfg.get("name", "centaurdrug-admet-panel")
    model_root = resolve_project_path(
        panel_cfg.get("model_root", "models/admet_xgboost")
    )
    output_root = resolve_project_path(
        bundle_cfg.get("output_root", "model_bundles")
    )
    runtime_model_subdir = bundle_cfg.get(
        "runtime_model_subdir",
        "admet_xgboost",
    )
    bundle_version = version or utc_version()

    bundle_dir = output_root / bundle_name / bundle_version
    runtime_model_root = bundle_dir / runtime_model_subdir

    if bundle_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Bundle already exists: {bundle_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(bundle_dir)

    bundle_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(model_root, runtime_model_root)

    manifest = build_manifest(
        model_root=model_root,
        bundle_name=bundle_name,
        bundle_version=bundle_version,
        runtime_model_subdir=runtime_model_subdir,
    )
    manifest["bundle_dir"] = str(bundle_dir)

    manifest_text = json.dumps(manifest, indent=2)
    (bundle_dir / "manifest.json").write_text(
        manifest_text,
        encoding="utf-8",
    )
    (runtime_model_root / RUNTIME_MANIFEST).write_text(
        manifest_text,
        encoding="utf-8",
    )

    verification = verify_bundle(bundle_dir)

    return {
        "status": verification["status"],
        "bundle_dir": str(bundle_dir),
        "runtime_model_root": str(runtime_model_root),
        "manifest_path": str(bundle_dir / "manifest.json"),
        "bundle_name": bundle_name,
        "bundle_version": bundle_version,
        "n_files": len(manifest["files"]),
        "verification": verification,
    }


def verify_bundle(bundle_dir: str | Path) -> Dict[str, Any]:
    bundle_path = resolve_project_path(bundle_dir)
    manifest_path = bundle_path / "manifest.json"

    if not manifest_path.exists():
        return {
            "status": "error",
            "bundle_dir": str(bundle_path),
            "errors": [f"missing manifest: {manifest_path}"],
        }

    manifest = read_json(manifest_path)
    runtime_model_subdir = manifest.get(
        "runtime_model_subdir",
        "admet_xgboost",
    )
    runtime_model_root = bundle_path / runtime_model_subdir
    readiness = validate_model_artifacts(runtime_model_root)

    errors = []

    if readiness["status"] != "ok":
        errors.append(f"incomplete model panel: {readiness['missing']}")

    for record in manifest.get("files", []):
        path = runtime_model_root / record["path"]

        if not path.exists():
            errors.append(f"missing file: {record['path']}")
            continue

        if path.stat().st_size != int(record["size_bytes"]):
            errors.append(f"size mismatch: {record['path']}")
            continue

        if sha256_file(path) != record["sha256"]:
            errors.append(f"sha256 mismatch: {record['path']}")

    return {
        "status": "ok" if not errors else "error",
        "bundle_dir": str(bundle_path),
        "runtime_model_root": str(runtime_model_root),
        "bundle_name": manifest.get("bundle_name"),
        "bundle_version": manifest.get("bundle_version"),
        "n_files": len(manifest.get("files", [])),
        "errors": errors,
    }


def register_bundle(
    *,
    config: Dict[str, Any],
    bundle_dir: str | Path,
    alias: str | None = None,
) -> Dict[str, Any]:
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    verification = verify_bundle(bundle_dir)

    if verification["status"] != "ok":
        raise RuntimeError(
            "Cannot register invalid bundle: "
            f"{verification['errors']}"
        )

    bundle_path = resolve_project_path(bundle_dir)
    manifest = read_json(bundle_path / "manifest.json")
    mlflow_cfg = config.get("mlflow", {})
    experiment_name = mlflow_cfg.get(
        "experiment_name",
        "centaurdrug-model-delivery",
    )
    registered_model_name = mlflow_cfg.get(
        "registered_model_name",
        manifest.get("bundle_name", "centaurdrug-admet-panel"),
    )
    artifact_path = mlflow_cfg.get("artifact_path", "model_bundle")

    mlflow.set_experiment(experiment_name)
    client = MlflowClient()

    try:
        client.create_registered_model(registered_model_name)
    except MlflowException as exc:
        if "already exists" not in str(exc).lower():
            raise

    with mlflow.start_run(
        run_name=(
            f"{registered_model_name}-"
            f"{manifest.get('bundle_version', 'unversioned')}"
        )
    ) as run:
        mlflow.log_params(
            {
                "bundle_name": manifest.get("bundle_name"),
                "bundle_version": manifest.get("bundle_version"),
                "bundle_schema_version": manifest.get("schema_version"),
                "git_commit": (manifest.get("git") or {}).get("commit"),
                "git_dirty": (manifest.get("git") or {}).get("dirty"),
            }
        )
        mlflow.log_artifacts(str(bundle_path), artifact_path=artifact_path)

        source = f"{run.info.artifact_uri.rstrip('/')}/{artifact_path}"
        model_version = client.create_model_version(
            name=registered_model_name,
            source=source,
            run_id=run.info.run_id,
            tags={
                "bundle_version": str(manifest.get("bundle_version")),
                "bundle_name": str(manifest.get("bundle_name")),
            },
        )

    if alias:
        client.set_registered_model_alias(
            registered_model_name,
            alias,
            model_version.version,
        )

    return {
        "status": "ok",
        "registered_model_name": registered_model_name,
        "model_version": model_version.version,
        "alias": alias,
        "run_id": model_version.run_id,
        "source": model_version.source,
    }


def print_json(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create, verify, and register CentaurDrug model bundles."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("--version", default=None)
    bundle_parser.add_argument("--overwrite", action="store_true")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--bundle-dir", required=True)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("--bundle-dir", required=True)
    register_parser.add_argument("--alias", default=None)

    args = parser.parse_args()
    config = load_delivery_config(args.config)

    if args.command == "bundle":
        print_json(
            create_bundle(
                config=config,
                version=args.version,
                overwrite=args.overwrite,
            )
        )
        return

    if args.command == "verify":
        print_json(verify_bundle(args.bundle_dir))
        return

    if args.command == "register":
        print_json(
            register_bundle(
                config=config,
                bundle_dir=args.bundle_dir,
                alias=args.alias,
            )
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
