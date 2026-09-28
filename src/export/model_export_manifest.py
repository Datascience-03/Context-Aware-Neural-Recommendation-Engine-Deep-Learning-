from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "data" / "processed" / "model"
EXPORT_DIR = PROJECT_ROOT / "data" / "processed" / "model_export"


def file_info(path: Path):
    if not path.exists():
        return {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "exists": False,
        }

    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": True,
        "size_bytes": path.stat().st_size,
    }


def create_manifest():
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {
        "project": "Context-Aware Neural Recommendation Engine",
        "member": "Member 1",
        "task": "Model Export",
        "day": 1,
        "purpose": (
            "Prepare and verify artifacts required for exporting "
            "trained Query and Candidate model architectures and weights."
        ),
        "model_components": {
            "query_tower": {
                "source": "src/feature_engineering/query_tower.py",
                "architecture_export": "query_tower.json",
                "weights_export": "query_tower.weights.h5",
            },
            "candidate_tower": {
                "source": "src/model/candidate_tower.py",
                "architecture_export": "candidate_tower.json",
                "weights_export": "candidate_tower.weights.h5",
            },
        },
        "existing_trained_artifacts": {
            "final_model_weights": file_info(
                MODEL_DIR / "final_weights.weights.h5"
            ),
            "best_model_weights": file_info(
                MODEL_DIR / "best_weights.weights.h5"
            ),
        },
        "status": "DAY 1 EXPORT PREPARATION COMPLETED",
    }

    output_path = EXPORT_DIR / "model_export_manifest.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("MODEL EXPORT MANIFEST")
    print("=" * 50)
    print(f"Project       : {manifest['project']}")
    print(f"Member        : {manifest['member']}")
    print(f"Task          : {manifest['task']}")
    print(f"Day           : {manifest['day']}")
    print()

    for name, artifact in manifest["existing_trained_artifacts"].items():
        print(f"{name}:")
        print(f"  Exists      : {artifact['exists']}")
        if artifact["exists"]:
            print(f"  Size        : {artifact['size_bytes']} bytes")

    print()
    print(f"Manifest saved: {output_path}")
    print()
    print("DAY 1 MODEL EXPORT PREPARATION PASSED")


if __name__ == "__main__":
    create_manifest()