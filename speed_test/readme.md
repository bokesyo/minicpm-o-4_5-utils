# MiniCPM-o 4.5 Duplex Omni Speed Test

## Download eval dataset

```bash
hf download bokesyo/omni_duplex_eval --repo-type=dataset --local-dir /root/omni_duplex_eval
```

## Download model

```bash
hf download openbmb/MiniCPM-o-4_5 --local-dir /root/MiniCPM-o-4_5
```

## Run inference
/root/miniconda3/bin/python ./eval_duplex_batch.py --video_dir /root/omni_duplex_eval/omni_demo_zh_duplex --output_dir ./eval_report

## Make data viewer
/root/miniconda3/bin/python ./data_viewer.py --results ./eval_report/eval_results.json