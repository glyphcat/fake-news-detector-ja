from test.helpers.script_runner import run_script


def _run_finetune_evaluate():
    try:
        run_script("モデル評価 Phase 2", "python src/evaluate.py --phase finetune")
        return 0
    except Exception as e:
        print(f"\n⛔️ エラー発生: {e}")
        return 1


def test_finetune_evaluate():
    result = _run_finetune_evaluate()
    assert result == 0
