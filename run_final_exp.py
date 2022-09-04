import multiprocessing as mp
import subprocess
import sys
import time
from typing import List
import torch

# Config
LOG_WANDB = "true"
MAX_JOBS = torch.cuda.device_count()
WANDB_PROJECT = "GRACE_recoverability_final"


def run_job(job: str) -> None:
    subprocess.call(job, shell=True)


def get_jobs() -> List[str]:
    prefix = f"{sys.executable} train.py use_wandb={LOG_WANDB} wandb_project={WANDB_PROJECT} "

    jobs = []

    # GRACE - Reddit
    jobs.append(f"dataset=Reddit2 method=GRACE Reddit2.eval_method=DGI exp_type=unsupervised")
    jobs.append(f"dataset=Reddit method=GRACE Reddit.eval_method=DGI exp_type=unsupervised")
    jobs.append(f"dataset=ogbn_products method=recoverability ogbn_products.eval_method=DGI exp_type=unsupervised")
    jobs.append(f"dataset=ogbn_arxiv method=recoverability ogbn_arxiv.eval_method=DGI exp_type=unsupervised")

    # Recoverability + Random + Supervised
    for exp_type in ("unsupervised", "random", "supervised"):
        jobs.append(f"dataset=ogbn_products method=recoverability ogbn_products.eval_method=DGI exp_type={exp_type}")
        jobs.append(f"dataset=ogbn_arxiv method=recoverability ogbn_arxiv.eval_method=DGI exp_type={exp_type}")
        jobs.append(f"dataset=Reddit2 method=recoverability Reddit2.eval_method=DGI exp_type={exp_type}")
        jobs.append(f"dataset=Reddit method=recoverability Reddit.eval_method=DGI exp_type={exp_type}")
        jobs.append(f"dataset=Cora method=recoverability Cora.eval_method=DGI exp_type={exp_type}")
        jobs.append(f"dataset=amazon_photos method=recoverability amazon_photos.eval_method=DGI exp_type={exp_type}")
        jobs.append(f"dataset=PPI method=recoverability PPI.eval_method=DGI exp_type={exp_type}")
        jobs.append(f"dataset=CiteSeer method=recoverability CiteSeer.eval_method=DGI exp_type={exp_type}")
        jobs.append(f"dataset=PubMed method=recoverability PubMed.eval_method=DGI exp_type={exp_type}")
        jobs.append(f"dataset=DBLP method=recoverability DBLP.eval_method=DGI exp_type={exp_type}")

    jobs = [prefix + j for j in jobs]


    return jobs


if __name__ == "__main__":
    jobs = get_jobs()
    print("Running jobs")
    for i, job in enumerate(jobs):
        print(f"Job ({i}): {job}")
    print(f"Total number of jobs: {len(jobs)}")

    running_procs: List[mp.Process] = []

    for curr_job in jobs:
        while len(running_procs) >= MAX_JOBS:
            done_process_idx = -1
            for i, curr_prop in enumerate(running_procs):
                curr_prop.join(0)
                if not curr_prop.is_alive():
                    # Process is done
                    done_process_idx = i
                    break

            if done_process_idx >= 0:
                running_procs.pop(done_process_idx)

            time.sleep(2)  # Not to kill the CPU

        proc = mp.Process(target=run_job, args=(curr_job,))
        proc.start()
        if MAX_JOBS > 1:
            time.sleep(60)  # Give time to allocate memory in GPU
        running_procs.append(proc)
