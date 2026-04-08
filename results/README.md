# Results

実行結果を集約するためのディレクトリです。

`legacy` 側の挙動互換を保つため、CLI のデフォルト出力先は変更していません。結果をこのディレクトリへまとめる場合は、各コマンドで出力先を明示してください。

## CNN 学習・評価

```bash
PYTHONPATH=src python3 -m apk2img_ml train-eval-mrun \
  --data-root ./images256 \
  --model tiny \
  --epochs 15 \
  --batch 32 \
  --workers 4 \
  --in-ch 1 \
  --resize 256,256 \
  --seed 3407 \
  --runs 3 \
  --log-dir ./results/train_eval_mrun
```

出力される主なファイル:

- `train_log.json`
- `lr_curves.png`
- `loss_curves.png`
- `val_acc_curves.png`

## 推奨配置

- `results/train_eval_mrun/`: CNN の学習・評価ログ
- `results/doc2vec/`: Doc2Vec 学習・推論の結果
- `results/images/`: ベクトル画像化後の確認用出力
- `results/notebooks/`: 実行結果の集計・可視化ノートブック
