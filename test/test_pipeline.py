from test.helpers.script_runner import run_script


def _run_all():
    try:
        run_script("データ前処理", "python src/prepare_data.py --test")
        run_script("データセット分割処理", "python src/split_data.py")
        run_script("モデル学習 Phase 1", "python src/train.py --phase pretune")
        run_script("モデル評価 Phase 1", "python src/evaluate.py --phase pretune")
        run_script("モデル学習 Phase 2", "python src/train.py --phase finetune")
        run_script("モデル評価 Phase 2", "python src/evaluate.py --phase finetune")
        print(f"\n✅ 全処理が正常に終了")
        return 0
    except Exception as e:
        print(f"\n⛔️ エラー発生: {e}")
        return 1


def test_pipeline():
    result = _run_all()
    assert result == 0
