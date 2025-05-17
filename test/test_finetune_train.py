from test.helpers.script_runner import run_script


def _run_finetune_train():
    try:
        run_script("モデル学習 Phase 2", "python src/train.py --phase finetune")
        return 0
    except Exception as e:
        print(f"\n⛔️ エラー発生: {e}")
        return 1


def test_finetune_train():
    result = _run_finetune_train()
    assert result == 0
