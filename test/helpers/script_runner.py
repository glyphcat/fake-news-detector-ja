import os
import subprocess

ENV = os.environ.copy()
ENV["PYTHONPATH"] = "."


def run_script(script_title: str, cmd: str):
    print(f"\n🚀 {script_title} 実行中...")
    # result = subprocess.run([sys.executable, target], check=True, env=ENV)
    result = subprocess.run(cmd, shell=True, check=True, env=ENV)
    if result.returncode != 0:
        raise RuntimeError(f"❌ {script_title} failed.")
    print(f"\n💎 {script_title} 完了")
