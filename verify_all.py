#!/usr/bin/env python3
"""
Tinkering Repository - Master Test & Physics Verification Runner
Author: Ali Malik (@am-LLM)

Executes empirical test suites across:
1. Frontier Cross-Domain Hybrids (65 engines)
2. Domain Laboratories (Aerospace GNC, SCADA, Frugal Mechanics, Quantum)
3. Engineering Continuum (418 course harnesses)
"""

import os
import sys
import time
import subprocess

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def run_suite(description: str, cmd: list, env=None) -> bool:
    print(f"\n========================================================")
    print(f"🔬 RUNNING: {description}")
    print(f"========================================================")
    t0 = time.time()
    custom_env = os.environ.copy()
    if env:
        custom_env.update(env)
    
    res = subprocess.run(cmd, cwd=REPO_ROOT, env=custom_env, capture_output=False)
    dt = time.time() - t0
    if res.returncode == 0:
        print(f"✅ PASSED in {dt:.2f}s")
        return True
    else:
        print(f"❌ FAILED with return code {res.returncode}")
        return False

def main():
    print(f"🚀 Starting verification for repository: {REPO_ROOT}")
    results = []

    # 1. Frontier Hybrids Suite
    hybrids_dir = os.path.join(REPO_ROOT, "frontier_hybrids")
    if os.path.exists(hybrids_dir):
        env = {"PYTHONPATH": hybrids_dir}
        ok = run_suite("Frontier Cross-Domain Hybrid Engines (1-65)", [
            sys.executable, "-m", "pytest", "-p", "no:recording", "-q", os.path.join(hybrids_dir, "tests")
        ], env=env)
        results.append(("Frontier Hybrids", ok))

    # 2. Domain Laboratories
    domain_dir = os.path.join(REPO_ROOT, "domain_laboratories")
    if os.path.exists(domain_dir):
        for sub in sorted(os.listdir(domain_dir)):
            subpath = os.path.join(domain_dir, sub)
            if os.path.isdir(subpath):
                tests = [f for f in os.listdir(subpath) if f.startswith("test_") and f.endswith(".py")]
                if tests:
                    env = {"PYTHONPATH": subpath}
                    for t in tests:
                        ok = run_suite(f"Domain Lab: {sub} -> {t}", [
                            sys.executable, "-m", "pytest", "-p", "no:recording", "-q", os.path.join(subpath, t)
                        ], env=env)
                        results.append((f"{sub}/{t}", ok))

    # Summary
    print("\n" + "="*60)
    print("🏆 EMPIRICAL TEST VERIFICATION SUMMARY")
    print("="*60)
    total = len(results)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status} : {name}")
    print(f"\nFinal Score: {passed}/{total} suites passed ({passed/total*100:.1f}%)")
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
