from test.helpers.script_runner import run_script


def _run_pretune_train():
    try:
        run_script("モデル学習 Phase 1", "python src/train.py --phase pretune")
        return 0
    except Exception as e:
        print(f"\n⛔️ エラー発生: {e}")
        return 1


def test_pretune_train():
    result = _run_pretune_train()
    assert result == 0
