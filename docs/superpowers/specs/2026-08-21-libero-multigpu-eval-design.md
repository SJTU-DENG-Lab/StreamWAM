# LIBERO Configurable Multi-GPU Evaluation Design

Add a Python multi-GPU manager for configurable physical GPU IDs, suites, and trial counts. The atomic workload is `(suite, task_id, trial_id)`. The ordered workload is split into contiguous balanced ranges so each GPU receives either `floor(N/G)` or `ceil(N/G)` trials; task splitting occurs only at range boundaries. Each GPU runs one worker process, exposes only its assigned physical GPU through `CUDA_VISIBLE_DEVICES`, loads the model once, and evaluates all assignments in its manifest.

Worker outputs and logs are isolated under the manager output directory. Workers suppress their final timing display. After every worker exits successfully, the manager merges task/trial outcomes and chunk-weighted timing into one top-level `results.json`, then prints one global summary. Existing single-GPU rollout behavior remains supported.

The default complete benchmark consists of `libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`, each with task IDs 0-9. GPU IDs, suites, and number of trials are CLI parameters. The manager terminates remaining workers and reports log locations when any worker fails.
