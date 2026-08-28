"""One-process-per-job RoboTwin simulator runner."""

from __future__ import annotations

import argparse
import faulthandler
import importlib
import json
import os
import signal
import sys
import traceback
from pathlib import Path

import yaml


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", required=True)
    parser.add_argument("--robotwin-home", required=True)
    parser.add_argument("--policy-dir", required=True)
    parser.add_argument("--server-port", type=int, required=True)
    parser.add_argument("--inference-mode", choices=("baseline", "cd", "ac-stream"), required=True)
    parser.add_argument("--replan-steps", type=int, required=True)
    parser.add_argument("--job-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--status-output", required=True)
    parser.add_argument("--prewarm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-tag", default="streamingwam")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_single_job(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RoboTwin job file must contain exactly one job object")
    required = {"task", "config", "trial"}
    if not required.issubset(payload):
        raise ValueError(f"RoboTwin job is missing fields: {sorted(required - set(payload))}")
    return payload


def write_job_status(path: Path, job: dict, *, phase: str, **extra) -> None:
    _atomic_json(path, {**job, "phase": str(phase), **extra})


def main() -> int:
    args = _parser().parse_args()
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    robotwin = Path(args.robotwin_home).resolve()
    policy_dir = Path(args.policy_dir).resolve()
    output = Path(args.output).resolve()
    status_output = Path(args.status_output).resolve()
    job = load_single_job(Path(args.job_file))
    write_job_status(status_output, job, phase="starting", gpu_id=args.gpu_id)
    os.chdir(robotwin)
    sys.path[:0] = [
        str(policy_dir), str(robotwin), str(robotwin / "script"),
        str(robotwin / "description" / "utils"),
    ]
    from eval_policy import class_decorator, eval_policy, get_camera_config, get_embodiment_config

    policy_module = importlib.import_module("deploy_policy")
    model = policy_module.get_model({
        "server_host": "127.0.0.1",
        "server_port": args.server_port,
        "inference_mode": args.inference_mode,
        "replan_steps": args.replan_steps,
        "prewarm": args.prewarm,
    })
    write_job_status(status_output, job, phase="connected", gpu_id=args.gpu_id)

    def environment_args(task: str, config: str) -> dict:
        with open(robotwin / "task_config" / f"{config}.yml", encoding="utf-8") as handle:
            values = yaml.safe_load(handle)
        values.update(
            task_name=task, task_config=config, ckpt_setting=args.checkpoint_tag,
            policy_name="deploy_policy", eval_video_log=False, eval_video_save_dir=None,
        )
        with open(robotwin / "task_config" / "_embodiment_config.yml", encoding="utf-8") as handle:
            embodiments = yaml.safe_load(handle)
        embodiment = values["embodiment"]
        if len(embodiment) == 1:
            robot_file = embodiments[embodiment[0]]["file_path"]
            values.update(left_robot_file=robot_file, right_robot_file=robot_file, dual_arm_embodied=True)
        elif len(embodiment) == 3:
            values.update(
                left_robot_file=embodiments[embodiment[0]]["file_path"],
                right_robot_file=embodiments[embodiment[1]]["file_path"],
                embodiment_dis=embodiment[2], dual_arm_embodied=False,
            )
        else:
            raise ValueError(f"unexpected embodiment configuration: {embodiment}")
        values["left_embodiment_config"] = get_embodiment_config(values["left_robot_file"])
        values["right_embodiment_config"] = get_embodiment_config(values["right_robot_file"])
        camera = get_camera_config(values["camera"]["head_camera_type"])
        values["head_camera_h"], values["head_camera_w"] = camera["h"], camera["w"]
        return values

    task, config, trial = job["task"], job["config"], int(job["trial"])
    model.current_task, model.current_config = task, config
    write_job_status(status_output, job, phase="environment_and_policy", gpu_id=args.gpu_id)
    try:
        task_environment = class_decorator(task)
        _, successes = eval_policy(
            task, task_environment, environment_args(task, config), model,
            100000 * (1 + args.seed) + trial,
            test_num=1, video_size=None, instruction_type="unseen",
            skip_get_obs_within_replan=False,
        )
        result = {
            **job, "status": "completed", "success": int(successes),
            "episodes": 1, "error": None,
        }
        return_code = 0
    except Exception as error:
        traceback.print_exc()
        result = {
            **job, "status": "infrastructure_error", "success": None,
            "episodes": 0, "error": repr(error),
        }
        return_code = 1
    finally:
        policy_module._finish_episode(model)
        model.close()
    timing = list(model._timing_records) + list(model._episode_records)
    _atomic_json(output, {"gpu_id": args.gpu_id, "result": result, "timing": timing})
    write_job_status(
        status_output, job, phase="complete", gpu_id=args.gpu_id,
        result_status=result["status"],
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
