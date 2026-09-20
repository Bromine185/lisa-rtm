"""Hyperbolic on-demand client, with the safety rails a money-spending tool needs.

    python fast/hyperbolic.py balance
    python fast/hyperbolic.py options
    python fast/hyperbolic.py list
    python fast/hyperbolic.py plan --arms 8            # dry run: what it would cost, creates nothing
    python fast/hyperbolic.py reap                     # terminate EVERYTHING. the panic button.

API key from $HYPERBOLIC_API_KEY, or --key-file (default ~/.hyperbolic_key, mode 600).

THE FAILURE MODE THIS FILE EXISTS TO PREVENT is not a bad API call, it is a forgotten instance. Eight
rentals at $1.79/hr bill $14.32 every hour whether or not anything is training, and the account
cannot overdraft -- `maxOverdraftCents` is 0, so it stops dead at the balance, mid-step, and a run
without resume state loses everything it did. So:

  - every create() is journalled to a state file BEFORE the request is sent, so a launcher that dies
    between "created" and "recorded" still leaves a trace for reap() to find
  - create() refuses without confirm=True and without a cost estimate
  - reap() terminates every active rental, not just the ones in the journal, because the journal can
    only ever be a lower bound
  - nothing here starts training; it provisions and tears down. Training is fast/train_arm.py.

Endpoints verified live on 2026-09-20:
  GET  /v2/customer/balance                      -> {"balanceCents":N,"maxOverdraftCents":0}
  GET  /v2/on-demand/rental-options               -> [{gpuType,gpuCount,costPerHourCents,nodes,...}]
  GET  /v2/on-demand/virtual-machine-rentals      -> [ ... ]
  POST /v2/on-demand/virtual-machine-rentals      -> create
  POST /v2/on-demand/virtual-machine-rentals/terminate
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.error
import urllib.request

BASE = "https://api.hyperbolic.ai/v2"
GPU_TYPE = "rtx-6000-pro"
REGION = "us-east-3"
STATE = pathlib.Path(os.environ.get("HYPERBOLIC_STATE", pathlib.Path.home() / ".hyperbolic_rentals.jsonl"))

# Cost-law constants (notes/2026-09-20-arm-allocation.md). Rows per arm-step at batch 64.
MROWS = {"det": 3.072, "es": 6.144}
LAW_A, LAW_B = 9.0, 22.9367          # t(ms) = A + B * Mrows; intercept is degenerate, see plan 0.2
STEPS = 104_950
ACCOUNT_WALL_USD = 101.0


def _key(path=None):
    k = os.environ.get("HYPERBOLIC_API_KEY")
    if k:
        return k.strip()
    p = pathlib.Path(path or pathlib.Path.home() / ".hyperbolic_key")
    if p.exists():
        return p.read_text().strip()
    sys.exit(f"no API key: set $HYPERBOLIC_API_KEY or put it in {p}")


class Hyperbolic:
    def __init__(self, key=None, base=BASE, timeout=60, dry_run=False):
        self.key, self.base, self.timeout, self.dry_run = key or _key(), base, timeout, dry_run

    def _req(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        # The WAF in front of the API rejects Python's default User-Agent with HTTP 403 /
        # Cloudflare error 1010 (a browser-signature ban). curl passes, urllib does not, so send a
        # UA explicitly -- verified 2026-09-20: no UA fails, "curl/8.5.0" and "Mozilla/5.0" succeed.
        r = urllib.request.Request(f"{self.base}{path}", data=data, method=method,
                                   headers={"Authorization": f"Bearer {self.key}",
                                            "Content-Type": "application/json",
                                            "User-Agent": "lisa-rtm-fast/1.0 (curl-compatible)"})
        try:
            with urllib.request.urlopen(r, timeout=self.timeout) as f:
                raw = f.read().decode()
            return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:400]
            raise RuntimeError(f"{method} {path} -> HTTP {e.code}: {detail}") from None

    # ---- reads (free, safe) ------------------------------------------------------------------
    def balance_usd(self):
        b = self._req("GET", "/customer/balance")
        return b["balanceCents"] / 100.0, b.get("maxOverdraftCents", 0) / 100.0

    def options(self):
        return self._req("GET", "/on-demand/rental-options")

    def rentals(self):
        return self._req("GET", "/on-demand/virtual-machine-rentals") or []

    def option_for(self, gpu_count, gpu_type=GPU_TYPE, region=REGION):
        for o in self.options():
            if (o["gpuType"] == gpu_type and o["gpuCount"] == gpu_count
                    and o.get("region") == region and o.get("enabled")):
                return o
        return None

    # ---- writes (spend money) ----------------------------------------------------------------
    def _journal(self, event, payload):
        STATE.parent.mkdir(parents=True, exist_ok=True)
        with STATE.open("a") as f:
            f.write(json.dumps({"t": int(time.time()), "event": event, **payload}) + "\n")

    def create(self, label, gpu_count=1, gpu_type=GPU_TYPE, region=REGION, confirm=False):
        """Create one rental. Refuses without confirm=True. Journalled before the request goes out,
        so a crash between send and response still leaves something reap() can find."""
        opt = self.option_for(gpu_count, gpu_type, region)
        if opt is None:
            raise RuntimeError(f"no enabled option for {gpu_type} x{gpu_count} in {region}")
        body = {"gpuType": gpu_type, "gpuCount": gpu_count, "region": region, "label": label}
        est = opt["costPerHourCents"] / 100.0
        if self.dry_run:
            return {"dry_run": True, "body": body, "usd_per_hour": est}
        if not confirm:
            raise RuntimeError("create() requires confirm=True -- this spends money")
        self._journal("create_attempt", {"label": label, "body": body, "usd_per_hour": est})
        r = self._req("POST", "/on-demand/virtual-machine-rentals", body)
        self._journal("created", {"label": label, "response": r})
        return r

    def terminate(self, rental_id):
        if self.dry_run:
            return {"dry_run": True, "id": rental_id}
        r = self._req("POST", "/on-demand/virtual-machine-rentals/terminate", {"id": rental_id})
        self._journal("terminated", {"id": rental_id})
        return r

    def reap(self, confirm=False):
        """Terminate every active rental. The journal is only ever a lower bound on what exists, so
        this works from the live list instead."""
        live = self.rentals()
        out = []
        for r in live:
            rid = r.get("id") or r.get("rentalId")
            if not rid:
                continue
            if not confirm and not self.dry_run:
                out.append({"id": rid, "skipped": "needs confirm"})
                continue
            try:
                out.append({"id": rid, "result": self.terminate(rid)})
            except Exception as e:
                out.append({"id": rid, "error": str(e)[:200]})
        return out


# ---- cost planning (pure, testable, spends nothing) -------------------------------------------
def arm_step_ms(kind, a=LAW_A, b=LAW_B, factor=1.0):
    return (a + b * MROWS["det" if kind.startswith("det") else "es"]) * factor


def plan_cost(arms, usd_per_gpu_hr=1.79, steps=STEPS, factor=1.0, stage_hours=0.3):
    """Eight parallel single-GPU rentals, each billed until ITS arm finishes.

    Billing is per rental and wall-clock, so the det arms -- at roughly half an ES arm's step time --
    stop costing when they finish rather than idling until the slowest arm is done. That is the one
    real advantage of separate rentals over the 8-GPU node, which bills all eight for the max.
    """
    rows = []
    for name, (kind, _lam, _cls) in arms.items():
        ms = arm_step_ms(kind, factor=factor)
        h = steps * ms / 1000 / 3600 + stage_hours
        rows.append({"arm": name, "ms_per_step": round(ms, 1), "hours": round(h, 2),
                     "usd": round(h * usd_per_gpu_hr, 2)})
    total = round(sum(r["usd"] for r in rows), 2)
    wall = round(max(r["hours"] for r in rows), 2)
    node8 = round(wall * usd_per_gpu_hr * 8, 2)          # the 8-GPU SKU bills all eight for the max
    return {"rows": rows, "total_usd": total, "wall_clock_h": wall,
            "eight_gpu_node_usd": node8, "separate_rentals_save_usd": round(node8 - total, 2),
            "fits_account_wall": total < ACCOUNT_WALL_USD}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["balance", "options", "list", "plan", "reap"])
    ap.add_argument("--key-file")
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--factor", type=float, default=1.0, help="Blackwell factor from calibration")
    a = ap.parse_args()
    h = Hyperbolic(key=_key(a.key_file), dry_run=a.dry_run)

    if a.cmd == "balance":
        bal, over = h.balance_usd()
        print(f"balance ${bal:.2f}   max overdraft ${over:.2f}")
        if over == 0:
            print("  the account CANNOT overdraft: it stops dead at the balance, mid-step")
    elif a.cmd == "options":
        for o in h.options():
            n = (o.get("nodes") or [{}])[0]
            print(f"{o['gpuType']:<16} x{o['gpuCount']:<2} ${o['costPerHourCents'] / 100:>6.2f}/hr  "
                  f"{o.get('region'):<10} {n.get('vcpuCount')} vCPU {n.get('ramGb')} GB "
                  f"{n.get('storageGb')} GB  enabled={o.get('enabled')}")
    elif a.cmd == "list":
        live = h.rentals()
        print(f"{len(live)} active rental(s)")
        for r in live:
            print(" ", json.dumps(r)[:300])
        if not live:
            print("  nothing is billing")
    elif a.cmd == "plan":
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from fast.run_contract import ARMS
        p = plan_cost(ARMS, factor=a.factor)
        for r in p["rows"]:
            print(f"  {r['arm']:<18} {r['ms_per_step']:>7} ms/step  {r['hours']:>5} h  ${r['usd']:>6}")
        print(f"\n  total ${p['total_usd']}   wall-clock {p['wall_clock_h']} h")
        print(f"  8-GPU node would be ${p['eight_gpu_node_usd']} "
              f"(separate rentals save ${p['separate_rentals_save_usd']})")
        print(f"  fits the ${ACCOUNT_WALL_USD:.0f} account wall: {p['fits_account_wall']}")
    elif a.cmd == "reap":
        res = h.reap(confirm=a.confirm)
        print(json.dumps(res, indent=1) if res else "nothing to reap")
        if res and not a.confirm and not a.dry_run:
            print("\n  nothing was terminated -- re-run with --confirm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
