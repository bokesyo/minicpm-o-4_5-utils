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

```bash

/root/miniconda3/bin/python ./eval_duplex_batch.py --video_dir /root/omni_duplex_eval/omni_demo_zh_duplex --output_dir ./eval_report

/root/miniconda3/bin/python ./eval_duplex_batch.py --video_dir /root/omni_duplex_eval/omni_demo_zh_duplex --output_dir ./eval_report_w_tts_pad
```

## Make data viewer

```bash
/root/miniconda3/bin/python ./data_viewer.py --results ./eval_report/eval_results.json

/root/miniconda3/bin/python ./data_viewer.py --results ./eval_report_w_tts_pad/eval_results.json
```

## Make compiled data viewer

```bash
/root/miniconda3/bin/python ./data_viewer.py --results ./eval_report_compiled/eval_results.json
```

## Make compare viewer

```bash
/root/miniconda3/bin/python ./compare_viewer.py --results ./eval_report/eval_results.json --compiled_results ./eval_report_compiled/eval_results.json
```
