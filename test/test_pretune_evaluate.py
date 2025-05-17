from test.helpers.script_runner import run_script


def _run_pretune_evaluate():
    try:
        run_script("モデル評価 Phase 1", "python src/evaluate.py --phase pretune")
        return 0
    except Exception as e:
        print(f"\n⛔️ エラー発生: {e}")
        return 1


def test_pretune_evaluate():
    result = _run_pretune_evaluate()
    assert result == 0
