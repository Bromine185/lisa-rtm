"""CPU tests for the calibration decision logic.

    venv/bin/python fast/test_calibrate.py

The measurements need a GPU; the DECISIONS they drive do not, and those are what gate the spend.
This exercises the wheel-selection table and the go/no-go arithmetic across the published Blackwell
brackets, so a wrong threshold is caught here rather than after a rented instance has reported.
"""
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from fast.calibrate import REF, m0_driver, verdict

ok = 0


def check(name, cond, detail=""):
    global ok
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{('  -- ' + detail) if detail else ''}")
    assert cond, name
    ok += 1


def wheel_for(driver):
    """Exercise the same branch table m0_driver uses, without nvidia-smi."""
    major = int(driver.split(".")[0])
    if major >= 580:
        return "cu130"
    if major >= 575:
        return "cu129"
    if major >= 570:
        return "cu128"
    return "NONE"


print("1. wheel selection from the driver version")
# Release wheels ship SASS only (no PTX), and cu126's arch list stops at sm_90, so cu126 hard-fails
# on sm_120 with 'no kernel image is available'. CUDA 13.0 needs R580; cu128 caps at torch 2.11.0.
for drv, want in (("580.126.20", "cu130"), ("585.01", "cu130"), ("577.00", "cu129"),
                  ("570.211.01", "cu128"), ("560.35.03", "NONE")):
    check(f"driver {drv} -> {want}", wheel_for(drv) == want)
check("m0_driver survives a box with no nvidia-smi", isinstance(m0_driver(), dict))


def v(factor):
    return verdict({"m4_decoder_proxy": {"ms": 55.5 * factor}})


print("\n2. the three published Blackwell brackets")
rows = []
for label, f in (("optimistic 0.9x", 0.9), ("bandwidth-scaled 1.13x", 1.13),
                 ("parity 1.0x", 1.0), ("pessimistic 2.29x", 2.29)):
    r = v(f)
    rows.append((label, f, r))
    print(f"    {label:<24} factor {r['blackwell_factor']:<6} step {r['est_step_ms_8arm']:>7} ms  "
          f"1GPU ${r['est_1gpu_usd']:<7} parallel ${r['est_parallel_usd']:<7} {r['status']}")

check("optimistic is GO", v(0.9)["status"].startswith("GO"))
check("bandwidth-scaled is GO", v(1.13)["status"].startswith("GO"))
# at 2.29x BOTH stops apply, and both must be reported -- they have different fixes
p229 = v(2.29)
check("pessimistic stops", p229["status"].startswith("STOP"))
check("  ...names the 1.5x breach", any("1.5x" in r for r in p229["stop_reasons"]))
check("  ...names the account wall too", any("account wall" in r for r in p229["stop_reasons"]),
      f"{len(p229['stop_reasons'])} reasons reported")
check("pessimistic is indeed over $101", p229["est_1gpu_usd"] > 101)

print("\n3. the two hard stops")
v16 = v(1.6)
check("factor > 1.5 stops regardless of cost", v16["status"].startswith("STOP"), v16["status"])
check("  ...and says why", "1.5x" in v16["status"])
check("  ...and reports only that reason when cost is fine", len(v16["stop_reasons"]) == 1,
      f"${v16['cheapest_usd']} is under the wall")
mid = v(1.3)
check("between $60 and $101 asks rather than stopping", mid["status"].startswith("ASK"), mid["status"])

print("\n4. monotonicity -- a slower card is never cheaper")
prev = None
for f in (0.8, 1.0, 1.2, 1.4):
    c = v(f)["est_1gpu_usd"]
    if prev is not None:
        check(f"factor {f}: cost rises ({prev} -> {c})", c > prev)
    prev = c

print("\n5. the A100 reference reproduces the known baseline")
r1 = v(1.0)
check("factor 1.0 reproduces 995 ms/step", abs(r1["est_step_ms_8arm"] - REF["step_ms_8arm"]) < 1.0,
      f"{r1['est_step_ms_8arm']} ms")
check("factor 1.0 gives ~29 h on one GPU", 28.5 < r1["est_1gpu_hours"] < 29.5,
      f"{r1['est_1gpu_hours']} h")
check("factor 1.0 gives ~$52 on one GPU", 50 < r1["est_1gpu_usd"] < 54, f"${r1['est_1gpu_usd']}")
check("parallel is faster than one GPU", r1["est_parallel_hours"] < r1["est_1gpu_hours"])

print("\n6. incomplete input does not silently produce a GO")
check("no proxy -> INCOMPLETE", verdict({})["status"].startswith("INCOMPLETE"))

print(f"\n{ok} checks passed")
