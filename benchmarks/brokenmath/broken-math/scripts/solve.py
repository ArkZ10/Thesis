from sycophancy.solve import solve
from sycophancy.configs import load_config
from sycophancy.run_manifest import select_files, ensure_manifest
from sycophancy.best_of_n import single, random_selector, dual
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
import os
import glob
import argparse
import re
from datetime import datetime
import json

def execute_judging_task(task_details):
    """
    Worker function to execute a single judging task.
    """
    project_config = task_details["project_config"]
    solver_config = task_details["solver_config"]
    files = task_details["files"]
    judge_part = task_details["judge_part"]

    mode = judge_part.get("mode")
    if mode in ["discrete", "continuous"]:
        single(project_config, solver_config, files, judge_part)
    elif mode == "random":
        random_selector(project_config, solver_config, files, judge_part)
    elif mode == "dual":
        dual(project_config, solver_config, files, judge_part)
    else:
        raise ValueError(f"Unknown judge mode: {mode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve solved problems using the batch API.")
    parser.add_argument("--project")
    parser.add_argument("--config-base", default="configs/projects")
    parser.add_argument("--synchronous", action='store_true')
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Run at most this many of the selected problems, before sharding")
    parser.add_argument("--manifest", help="Persist selection and configuration checksums for safe resume")
    parser.add_argument("--dry-run", action='store_true', help="Prepare selection without loading models")

    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error('Require 0 <= shard-index < num-shards')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')

    project_config = load_config(os.path.join(args.config_base, args.project + ".yaml"))

    solver_config = load_config(project_config.solver_config)

    models_to_run = project_config.model_configs

    all_problem_files = glob.glob(f"{project_config.unsolved_base_folder}/**/*.json", recursive=True)

    files_to_run = [
        file for file in all_problem_files 
        if any(re.match(regex, file.replace(project_config.unsolved_base_folder + "/", "")) for regex in solver_config.problem_regexes)
    ]

    if solver_config.earliest_date_added is not None:
        if " " not in solver_config.earliest_date_added:
            solver_config.earliest_date_added += " 00:00:00"
        date_threshold = datetime.strptime(solver_config.earliest_date_added, "%Y-%m-%d %H:%M:%S")
        files_to_actually_run = []
        for file in files_to_run:
            with open(file, "r") as f:
                data = json.load(f)
                if "date_added" in data and datetime.strptime(data["date_added"], "%Y-%m-%d %H:%M:%S") >= date_threshold:
                    files_to_actually_run.append(file)
    else:
        files_to_actually_run = files_to_run
    files_to_actually_run = select_files(files_to_actually_run, solver_config.n_samples,
                                        solver_config.sampling_seed)
    local_llada = any(load_config(os.path.join(project_config.model_config_folder, name + '.yaml')).get('api') == 'llada'
                      for name in models_to_run)
    manifest_path = args.manifest
    if local_llada and manifest_path is None:
        manifest_path = f'data/manifests/{args.project}.json'
    if manifest_path:
        manifest = ensure_manifest(manifest_path, project_config, solver_config, files_to_actually_run)
        print(f"Selection manifest: {manifest_path} ({len(manifest['problems'])} problems)")
    if args.limit is not None:
        files_to_actually_run = files_to_actually_run[:args.limit]
    files_to_actually_run = files_to_actually_run[args.shard_index::args.num_shards]

    print(f"Found {len(files_to_actually_run)} files to run.")
    print(f"Running {len(files_to_actually_run)} problems for project {args.project}")
    if args.dry_run:
        for file in files_to_actually_run:
            print(json.load(open(file))['problem_id'])
        raise SystemExit(0)
    if not files_to_actually_run:
        raise SystemExit(0)
    if args.synchronous:
        total_cost = 0
        for config_name in models_to_run:
            total_cost += solve(
                project_config, 
                solver_config, 
                files_to_actually_run, 
                config_name
            )
    else:
        with ThreadPoolExecutor(max_workers=len(models_to_run)) as executor:
            futures = []
            for config_name in models_to_run:
                futures.append(
                    executor.submit(
                        solve,
                        project_config,
                        solver_config,
                        files_to_actually_run,
                        config_name,
                    )
                )

            total_cost = 0
            for future in futures:
                total_cost += future.result()
    print(f"Total cost: {total_cost:.2f} USD")

    if solver_config.type == "best_of_n":
        files_to_judge = [
            file.replace(project_config.unsolved_base_folder, project_config.solved_base_folder)
            for file in files_to_actually_run
        ]

        # Prepare a list of all judging tasks to be executed
        judging_tasks = []
        for judge_part in solver_config.judges:
            for i, model in enumerate(models_to_run):
                if not judge_part.get("same_judge") and i > 0:
                    continue
                
                # Create a mutable copy of the judge configuration for each task
                current_judge_part = judge_part.copy()
                if current_judge_part.get("same_judge"):
                    current_judge_part["judge"] = model
                    current_judge_part["name"] = f"{model} {current_judge_part['name']}"

                task_details = {
                    "project_config": project_config,
                    "solver_config": solver_config,
                    "files": files_to_judge,
                    "judge_part": current_judge_part,
                }
                judging_tasks.append(task_details)
        # Execute the judging tasks in parallel using a ThreadPoolExecutor
        if judging_tasks:
            print(f"Running {len(judging_tasks)} judging tasks in parallel.")
            # Adjust max_workers as needed; here it's set to the number of tasks.
            with ThreadPoolExecutor(max_workers=len(judging_tasks)) as executor:
                futures = [executor.submit(execute_judging_task, task) for task in judging_tasks]
                # Wait for all judging tasks to complete
                for future in futures:
                    future.result()  # This will raise any exceptions that occurred in a thread

        for file in files_to_judge:
            data = json.load(open(file, "r"))
            if "attempts" not in data:
                continue
            # what we are going to do, is compress the attempts if the solution is the same
            attempts = data["attempts"]
            if len(attempts) == 0:
                continue
            compressed_attempts = []
            for attempt in attempts:
                to_add = True
                for i, existing_attempt in enumerate(compressed_attempts):
                    if existing_attempt["solution"] == attempt["solution"]:
                        to_add = False
                        if "other_selectors" not in existing_attempt:
                            existing_attempt["other_selectors"] = []
                        existing_attempt["other_selectors"].append(attempt)
        
                if to_add:
                    compressed_attempts.append(attempt)
            data["attempts"] = compressed_attempts
            with open(file, "w") as f:
                json.dump(data, f, indent=4)
