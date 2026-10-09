"""Pack the validated same-world B10 predictions for the diagnostic reference."""
import json
from pathlib import Path
import numpy as np
from safetensors import safe_open
from remote_train_heads import ROOT,sha

if __name__=="__main__":
    root=ROOT/"runs/target_diagnostic_2026-10-01";out=root/"B10-reference";out.mkdir(exist_ok=True)
    assert json.loads((root/"extract-status.json").read_text())["stage"]=="verified"
    index=list(map(json.loads,(root/"index.jsonl").read_text().splitlines()))
    rows=[r for r in index if r["split"]=="head_val"];poses=[]
    assert len(rows)==768
    for row in rows:
        with safe_open(row["shard_path"],framework="pt",device="cpu") as stored:
            position=row["row_index"];poses.append(stored.get_slice("teacher_poses")[position:position+1][0].numpy().copy())
    poses=np.stack(poses)
    path=out/"reference-predictions.npz"
    np.savez(path,scene_tokens=np.asarray([r["scene_token"] for r in rows]),generation_seeds=np.asarray([r["generation_seed"] for r in rows]),trajectories=poses,teacher_poses=poses)
    report={"EXTRACT_LOCK_sha256":sha(root/"EXTRACT_LOCK.json"),"prediction_sha256":sha(path),"rows":768,"distinct_scenes":256,"navtest_used":False}
    (out/"prepare-report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
