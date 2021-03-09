from pathlib import Path
from typing import Optional, Sequence, Tuple
import numpy as np


def parse_eff_line(s: str, decimals: Optional[None]) -> float:
    if decimals:
        return round(float(s.split("_")[-1]), decimals)
    return float(s.split("_")[-1])


def find_min_val_error(
    path: Path, filename: str = "val_error.log"
) -> Tuple[float, int]:
    f = path.joinpath(filename)
    txt = f.read_text()
    txt_spl = txt.split("\n")
    a = [float(i) for i in txt_spl if not i == ""]
    min_ = min(a)
    argmin = a.index(min_)

    return min_, argmin


def find_test_eff(
    path: Path, match_string: str, filename: str = "eff_results.txt", decimals: int = 2
) -> Tuple[Sequence[float], float]:
    f = path.joinpath(filename)
    txt = f.read_text()
    txt_spl = txt.split("\n")
    eff = [parse_eff_line(t, decimals) for t in txt_spl if match_string in t]

    return eff


p = Path(r"C:\Users\Balco\Downloads\mppt_results\bc\real")

dirs = [dir_ for dir_ in p.iterdir() if dir_.is_dir()]
bc_all = []
bc_means = []

for dir_ in dirs:
    val_error = find_min_val_error(dir_)
    expert_eff = find_test_eff(dir_, match_string="po-expert-test", decimals=2)
    bc_eff = find_test_eff(dir_, match_string="bc-agent-test", decimals=2)
    bc_all.append(bc_eff)

    print(dir_)
    print(f"Min val error: {val_error[0]}")
    print(f"Val checks: {val_error[1]}")
    for (day, exp) in enumerate(expert_eff, start=1):
        print(f"Expert eff day {day}: {exp}")
    print(f"Average expert eff: {round(np.mean(expert_eff), 2)}")
    print(f"Expert 2SD: {round(np.std(expert_eff)*2, 2)}")
    for (day, bc) in enumerate(bc_eff, start=1):
        print(f"BC eff day {day}: {bc}")
    print(f"Average bc eff: {round(np.mean(bc_eff), 2)}")
    print(f"BC 2SD: {round(np.std(bc_eff)*2, 2)}")
    print()


bc_all_mean = np.mean(bc_all, axis=0)
bc_all_std = np.std(bc_all, axis=0) * 2
for day, (mean_, std_) in enumerate(zip(bc_all_mean, bc_all_std), start=1):
    print(f"BC mean eff day {day}: {mean_}")
    print(f"BC 2SD day {day}: {round(np.mean(std_), 2)}")
