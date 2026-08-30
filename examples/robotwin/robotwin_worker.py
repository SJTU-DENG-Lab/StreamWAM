"""One-process-per-job RoboTwin simulator runner."""

from __future__ import annotations

import argparse
import faulthandler
import importlib.util
import json
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

import yaml


def _normalize_pci_bus_id(value: str) -> str:
    value = value.strip().lower()
    domain, bus, device_function = value.split(":")
    device, function = device_function.split(".", 1)
    return (
        f"{int(domain, 16):04x}:{int(bus, 16):02x}:"
        f"{int(device, 16):02x}.{int(function, 16):x}"
    )


def _sapien_render_device_alias(gpu_id: str) -> str:
    override = os.environ.get("STREAMINGWAM_SAPIEN_RENDER_DEVICE")
    if override and override.lower() != "auto":
        return override
    completed = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(gpu_id),
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return f"pci:{_normalize_pci_bus_id(completed.stdout.strip().splitlines()[0])}"


def _bind_sapien3_renderer(gpu_id: str) -> None:
    """Bind RoboTwin's Vulkan renderer to the requested physical GPU."""

    import sapien as sapien_package
    import sapien.core as sapien

    version = str(getattr(sapien_package, "__version__", ""))
    if not version.startswith("3."):
        return
    alias = _sapien_render_device_alias(gpu_id)
    device = sapien.Device(alias)
    if not device.can_render():
        raise RuntimeError(f"SAPIEN device {alias} cannot render")
    logged = False

    def create_renderer(**_kwargs):
        return sapien.render.SapienRenderer(device)

    def create_scene(_engine, config=None):
        nonlocal logged
        config = sapien.SceneConfig() if config is None else config
        sapien.physx.set_scene_config(config)
        scene = sapien.Scene(
            [sapien.physx.PhysxCpuSystem(), sapien.render.RenderSystem(device)]
        )
        selected = scene.render_system.device
        if selected.pci_string != device.pci_string:
            raise RuntimeError(
                "SAPIEN render device mismatch: "
                f"requested={device.pci_string}, selected={selected.pci_string}"
            )
        if not logged:
            print(
                f"[robotwin worker] renderer={selected.pci_string} "
                f"cuda_id={selected.cuda_id}",
                flush=True,
            )
            logged = True
        return scene

    sapien.SapienRenderer = create_renderer
    sapien_package.SapienRenderer = create_renderer
    sapien.Engine.create_scene = create_scene


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


def load_policy_module(policy_dir: Path):
    """Load this checkout's deploy policy without top-level-name ambiguity."""

    path = (policy_dir / "deploy_policy.py").resolve()
    if not path.is_file():
        raise FileNotFoundError(f"RoboTwin deploy policy does not exist: {path}")
    module_name = f"_streamingwam_robotwin_deploy_policy_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load RoboTwin deploy policy from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = (
        "get_model",
        "eval",
        "reset_model",
        "prepare_instruction",
        "prewarm_model",
        "begin_timing_trajectory",
        "end_timing_trajectory",
        "_finish_episode",
    )
    missing = [name for name in required if not callable(getattr(module, name, None))]
    if missing:
        raise AttributeError(
            f"deploy policy {path} is missing required hooks: {', '.join(missing)}"
        )
    print(f"[robotwin worker] deploy_policy={path}", flush=True)
    return module


def main() -> int:
    args = _parser().parse_args()
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)
    faulthandler.register(signal.SIGUSR1, file=sys.stderr, all_threads=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    _bind_sapien3_renderer(str(args.gpu_id))
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
    import eval_policy as official_eval
    from examples.robotwin.evaluator import evaluate_one_episode

    class_decorator = official_eval.class_decorator
    get_camera_config = official_eval.get_camera_config
    get_embodiment_config = official_eval.get_embodiment_config

    policy_module = load_policy_module(policy_dir)
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
        _, successes = evaluate_one_episode(
            official=official_eval,
            task_name=task,
            task_env=task_environment,
            settings=environment_args(task, config),
            model=model,
            start_seed=100000 * (1 + args.seed) + trial,
            trial=trial,
            instruction_type="unseen",
            policy_module=policy_module,
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
    for record in timing:
        record.setdefault("task", task)
        record.setdefault("config", config)
        record.setdefault("trial", trial)
    _atomic_json(output, {"gpu_id": args.gpu_id, "result": result, "timing": timing})
    write_job_status(
        status_output, job, phase="complete", gpu_id=args.gpu_id,
        result_status=result["status"],
    )
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
