import argparse
import pandas as pd
import wandb
from datetime import datetime


def make_hyperlink(url, value):
    return '=HYPERLINK("%s", "%s")' % (url, value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-name", type=str, required=True, help="The name of the W&B project (full path: <entity/project-name>) ")
    parser.add_argument("--output-csv", type=str, required=True, help="Where to store the results (CSV path)")

    args = parser.parse_args()
    proj_name = args.project_name
    output_path = args.output_csv
    if not output_path.endswith(".csv"):
        raise ValueError("Output CSV must end with '.csv'")

    api = wandb.Api()

    # Project is specified by <entity/project-name>
    runs = api.runs(proj_name)

    agg_summary = []

    for run in runs:
        if run.state != "finished":
            continue
        run_summary = {k: v for k, v in run.config.items()
                            if not k.startswith('_')}
        run_summary.update({k: v for k, v in run.summary._json_dict.items() if
                            isinstance(v, (int, float)) and not k.startswith("_")})

        agg_summary.append({"Dataset": run_summary["dataset"],
                            "Exp": run_summary["exp_type"],
                            "Method": run_summary["method"],
                            "F1Mi_AVG": run_summary['F1Mi_mean'],
                            "F1Mi_STD": run_summary['F1Mi_std'],
                            "F1Ma_AVG": run_summary['F1Ma_mean'],
                            "F1Ma_STD": run_summary['F1Ma_std'],
                            "Acc_AVG": run_summary['Accuracy_mean'],
                            "Acc_STD": run_summary['Accuracy_std'],
                            "URL": make_hyperlink(run.url, run.name)
                            })

    df = pd.DataFrame.from_dict(agg_summary)
    df = df.sort_values(["Method", "Exp", "Dataset"], ascending = (True, True, True))
    df.to_csv(output_path)