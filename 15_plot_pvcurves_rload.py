import os

import matlab.engine

from typing import Optional, Sequence, Tuple
from src.pv_array_dcdc import PVArray, PVSimResult
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np

COLOR_LIST = [
    "#1F77B4",  # muted blue
    "#FF7F0E",  # safety orange
    "#2CA02C",  # cooked asparagus green
    "#D62728",  # brick red
    "#9467BD",  # muted purple
    "#8C564B",  # chestnut brown
    "#E377C2",  # raspberry yogurt pink
    "#7F7F7F",  # middle gray
    "#BCBD22",  # curry yellow-green
    "#17BECF",  # blue-teal
]


plt.rcParams.update(
    {
        "axes.prop_cycle": plt.cycler(color=COLOR_LIST),
        "text.usetex": True,
        # "legend.loc": "upper right",
        # "font.family": "CMU Serif",
    }
)


PV_PARAMS_PATH = os.path.join("parameters", "01_pvarray.json")
PV_CKP_PATH = os.path.join("data", "01_pvarray_iv.json")
MODEL_NAME = "pv_boost_avg_rload"


def get_ivcurve(
    pvarray: PVArray, g: float = 1000, t: float = 25, points: int = 100
) -> Tuple[Sequence[float]]:
    i = []
    v = []
    p = []

    for dc in range(0, points + 1):
        res = pvarray.simulate(dc / points, irradiance=g, ambient_temp=t)
        i.append(res.pv_current)
        v.append(res.pv_voltage)
        p.append(res.pv_power)

    return v, i, p


def normalize(x: Sequence[float]) -> Sequence[float]:
    return [x_ / max(x) for x_ in x]


def rload_curve(v_seq: Sequence[float], point: PVSimResult) -> Sequence[float]:
    slope = point.pv_current / point.pv_voltage
    print(1 / slope)
    return [v * slope for v in v_seq]


def intersect(
    x: Sequence[float], y1: Sequence[float], y2: Sequence[float]
) -> Tuple[float]:
    difs = [abs(y1_ - y2_) for y1_, y2_ in zip(y1, y2)]
    idx = difs.index(min(difs))

    # return x[idx + 1], y1[idx + 1]
    return idx


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


def plot_curves(
    pvarray: PVArray, g1: int, g2: int, t1: float, t2: float, points: int = 100
) -> None:
    v1, i1, p1 = get_ivcurve(pvarray, g=g1, t=t1, points=points)
    voc1 = line_intersection(((v1[0], i1[0]), (v1[10], i1[10])), ((0, 0), (50, 0)))
    v1[0] = voc1[0]
    i1[0] = 0
    p1[0] = 0

    v2, i2, p2 = get_ivcurve(pvarray, g=g2, t=t2, points=points)
    voc2 = line_intersection(((v2[0], i2[0]), (v2[10], i2[10])), ((0, 0), (50, 0)))
    v2[0] = voc2[0]
    i2[0] = 0
    p2[0] = 0

    mpp1 = pvarray.get_true_mpp(irradiance=g1, ambient_temp=t1)
    mpp2 = pvarray.get_true_mpp(irradiance=g2, ambient_temp=t2)
    r1_curve = rload_curve(v1, mpp1)
    r2_curve = rload_curve(v1, mpp2)
    r1 = round(mpp1.pv_voltage / mpp1.pv_current, 2)
    r2 = round(mpp2.pv_voltage / mpp2.pv_current, 2)
    int1 = intersect(v1, r1_curve, i2)
    int2 = intersect(v1, r2_curve, i1)

    f = plt.figure()
    ax = f.add_subplot(111)

    ax.plot(
        v1,
        i1,
        label=f"Curve 1 - {g1} $\mathrm{{W/m^2}}$ @ {t1} $\mathrm{{^o}}$C",
        color=COLOR_LIST[0],
    )
    ax.plot(
        v2,
        i2,
        label=f"Curve 2 - {g2} $\mathrm{{W/m^2}}$ @ {t2} $\mathrm{{^o}}$C",
        color=COLOR_LIST[1],
    )
    ax.plot(v1, r1_curve, ":k", linewidth=0.7)
    ax.plot(v1, r2_curve, "--k", linewidth=0.7)
    ax.plot(mpp1.pv_voltage, mpp1.pv_current, "ok")
    ax.plot(mpp2.pv_voltage, mpp2.pv_current, "ok")
    ax.plot(v1[int1], r1_curve[int1], "^k")
    ax.plot(v1[int2], r2_curve[int2], "^k")
    ax.set_ylim(top=max([max(i1), max(i2)]) * 1.05)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    ax.yaxis.labelpad = 15
    # ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    # ax.tick_params(axis='y', which='major', pad=15)
    ax.legend(loc="upper left")
    # plt.tight_layout()

    ax.text(
        *(v1[int(points * 0.95)] + 0.1, r1_curve[int(points * 0.95)] + 0.2),
        f"$\mathrm{{Load_1}}$ = {r1} $\Omega$",
        fontsize=10,
        rotation=np.arctan2(r1_curve[0], v1[0]) * 180 - 5,
        rotation_mode="anchor",
        usetex=True,
    )

    ax.text(
        *(v1[int(points * 0.85)] + 0.1, r2_curve[int(points * 0.85)] + 0.1),
        f"$\mathrm{{Load_2}}$ = {r2} $\Omega$",
        fontsize=10,
        rotation=np.arctan2(r2_curve[0], v1[0]) * 180 - 1,
        rotation_mode="anchor",
        usetex=True,
    )

    ax.text(
        *(mpp1.pv_voltage * 1.03, mpp1.pv_current * 0.98),
        "$\mathrm{{MPP_1}}$",
        fontsize=10,
        usetex=True,
    )

    ax.text(
        *(mpp2.pv_voltage * 0.88, mpp2.pv_current * 1.06),
        "$\mathrm{{MPP_2}}$",
        fontsize=10,
        usetex=True,
    )
    ax.text(
        *(v1[int1] * 0.90, r1_curve[int1] * 1.04),
        "$\mathrm{{X_2}}$",
        fontsize=10,
        usetex=True,
    )
    ax.text(
        *(v1[int2] * 1.03, r2_curve[int2] * 0.96),
        "$\mathrm{{X_1}}$",
        fontsize=10,
        usetex=True,
    )

    f1 = plt.figure()
    ax1 = f1.add_subplot(111)

    ax1.plot(
        v1,
        p1,
        label=f"Curve 1 - {g1} $\mathrm{{W/m^2}}$ @ {t1} $\mathrm{{^o}}$C",
        color=COLOR_LIST[0],
    )
    ax1.plot(
        v2,
        p2,
        label=f"Curve 2 - {g2} $\mathrm{{W/m^2}}$ @ {t2} $\mathrm{{^o}}$C",
        color=COLOR_LIST[1],
    )

    ax1.plot(mpp1.pv_voltage, mpp1.pv_power, "ok")
    ax1.plot(mpp2.pv_voltage, mpp2.pv_power, "ok")
    ax1.plot(v1[int2], p1[int2], "^k")
    ax1.plot(v1[int1], 68.5, "^k")
    # plt.setp(ax.get_xticklabels(), visible=False)

    ax1.text(
        *(mpp1.pv_voltage * 1.04, mpp1.pv_power * 1.0),
        "$\mathrm{{MPP_1}}$",
        fontsize=10,
        usetex=True,
    )

    ax1.text(
        *(mpp2.pv_voltage * 0.95, mpp2.pv_power * 0.88),
        "$\mathrm{{MPP_2}}$",
        fontsize=10,
        usetex=True,
    )

    ax1.text(
        *(v1[int1] * 0.97, 68.5 * 1.08),
        "$\mathrm{{X_2}}$",
        fontsize=10,
        usetex=True,
    )

    ax1.text(
        *(v1[int2] * 1.025, p1[int2] * 0.975),
        "$\mathrm{{X_1}}$",
        fontsize=10,
        usetex=True,
    )

    ax1.legend(loc="upper left")
    ax1.set_xlabel("Voltage (V)")
    ax1.set_ylabel("Power (W)")
    # plt.tight_layout()

    pvarray.save()

    return f, f1


try:
    ENGINE.quit()
except NameError:
    pass
ENGINE = matlab.engine.connect_matlab()

pvarray = PVArray.from_json(
    path=PV_PARAMS_PATH,
    engine=ENGINE,
    model_name=MODEL_NAME,
)

f, f1 = plot_curves(pvarray, g1=800, g2=500, t1=25, t2=15, points=1000)
f.savefig("ivcurve_rload.pdf", bbox_inches="tight")
f1.savefig("pvcurve_rload.pdf", bbox_inches="tight")
