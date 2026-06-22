import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

cache_capacity = [x * 1024 for x in [2, 3, 4, 5, 6, 7, 8, 9]]
item_size = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
cache_policies = [
    "heuristic",
    "none",
    "split_non_redundant",
    # "split_non_redundant_rsu",
    "random",
]
model_path = (
    "/media/anda-network/hdd/khoi/mrat/runs/2026-06-08-16-38-41_w1_55_w2_45/model.pth"
)

MAX_PARALLEL = 8


def build_jobs():
    jobs = []
    for policy in cache_policies:
        for cache in cache_capacity:
            jobs.append(
                [
                    "python",
                    "eval.py",
                    "--model_path",
                    model_path,
                    "--episodes",
                    "100",
                    "--rsu_cache_capacity",
                    str(cache),
                    "--name",
                    f"cachecapa_{cache}_{policy}",
                    "--cache_policy",
                    policy,
                ]
            )
        for item in item_size:
            jobs.append(
                [
                    "python",
                    "eval.py",
                    "--model_path",
                    model_path,
                    "--episodes",
                    "100",
                    "--item_size",
                    str(item),
                    "--name",
                    f"itemsize_{item}_{policy}",
                    "--cache_policy",
                    policy,
                ]
            )
    return jobs


def run_job(cmd):
    name = cmd[cmd.index("--name") + 1]
    result = subprocess.run(cmd, capture_output=True, text=True)
    status = "OK" if result.returncode == 0 else f"FAIL ({result.returncode})"
    return name, status, result.stderr[-500:] if result.returncode else ""


if __name__ == "__main__":
    jobs = build_jobs()
    print(f"Launching {len(jobs)} jobs, {MAX_PARALLEL} at a time...\n")

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            name, status, err = future.result()
            print(f"[{i}/{len(jobs)}] {status:12} {name}")
            if err:
                print(f"    stderr: {err}")

    print("\nAll jobs finished.")
