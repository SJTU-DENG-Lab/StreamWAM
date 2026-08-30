"""Canonical one-episode RoboTwin evaluation used by the public benchmark."""

from __future__ import annotations

import importlib
import traceback
from typing import Any

import numpy as np


def evaluate_one_episode(
    *,
    official: Any,
    task_name: str,
    task_env: Any,
    settings: dict[str, Any],
    model: Any,
    start_seed: int,
    trial: int,
    instruction_type: str = "unseen",
    policy_module: Any | None = None,
) -> tuple[int, int]:
    """Run one accepted RoboTwin episode with canonical trajectory timing.

    Scene construction, expert validation, instruction preparation, and model
    warmup are outside the timer.  Timing begins immediately before the first
    policy observation and ends when the terminal action call returns.
    """

    if getattr(task_env, "eval_video_path", None) is not None:
        raise RuntimeError("benchmark timing requires eval_video_log=false")

    if policy_module is None:
        policy_module = importlib.import_module(str(settings["policy_name"]))
    reset_model = getattr(policy_module, "reset_model")
    prepare_instruction = getattr(policy_module, "prepare_instruction")
    prewarm_model = getattr(policy_module, "prewarm_model")
    begin_timing = getattr(policy_module, "begin_timing_trajectory")
    end_timing = getattr(policy_module, "end_timing_trajectory")

    task_env.suc = 0
    task_env.test_num = 0
    now_seed = int(start_seed)
    settings["eval_mode"] = True

    while True:
        render_freq = settings["render_freq"]
        settings["render_freq"] = 0
        try:
            task_env.setup_demo(
                now_ep_num=0,
                seed=now_seed,
                is_test=True,
                **settings,
            )
            episode_info = task_env.play_once()
            task_env.close_env()
        except official.UnStableError:
            task_env.close_env()
            now_seed += 1
            settings["render_freq"] = render_freq
            continue
        except Exception:
            traceback.print_exc()
            task_env.close_env()
            now_seed += 1
            settings["render_freq"] = render_freq
            continue

        if not (task_env.plan_success and task_env.check_success()):
            now_seed += 1
            settings["render_freq"] = render_freq
            continue

        settings["render_freq"] = render_freq
        try:
            task_env.setup_demo(
                now_ep_num=0,
                seed=now_seed,
                is_test=True,
                **settings,
            )
        except official.UnStableError:
            task_env.close_env()
            now_seed += 1
            continue

        descriptions = official.generate_episode_descriptions(
            settings["task_name"],
            [episode_info["info"]],
            1,
        )
        instruction = np.random.choice(descriptions[0][instruction_type])
        task_env.set_instruction(instruction=instruction)

        prepare_instruction(task_env, model)
        if bool(getattr(model, "needs_prewarm", False)):
            prewarm_model(task_env, model, task_env.get_obs())
        reset_model(model)

        metadata = {
            "task": str(task_name),
            "config": str(settings["task_config"]),
            "trial": int(trial),
            "accepted_seed": int(now_seed),
        }
        begin_timing(model, metadata)
        success = False
        while task_env.take_action_cnt < task_env.step_lim:
            observation = (
                task_env.get_obs()
                if bool(model.should_request_observation())
                else None
            )
            policy_module.eval(task_env, model, observation)
            if task_env.eval_success:
                success = True
                break
        end_timing(
            model,
            success,
            {
                **metadata,
                "environment_action_count": int(task_env.take_action_cnt),
                "environment_step_limit": int(task_env.step_lim),
            },
        )

        if success:
            task_env.suc = 1
        task_env.test_num = 1
        task_env.close_env()
        if task_env.render_freq:
            task_env.viewer.close()
        return now_seed + 1, int(success)
