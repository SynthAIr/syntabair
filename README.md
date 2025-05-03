
## Generators

### Copula

```bash
python scripts/generate_with_copula.py --train_data_path data/real/train.csv --output_path data/synthetic/copula.csv
```

### CTGAN

```bash
python scripts/train_ctgan.py --data_path data/real/train.csv --model_path models/ctgan.pkl --batch_size 500 --epochs 500
```

```bash
python scripts/generate_with_ctgan.py --model_path models/ctgan.pkl --output_path data/synthetic/ctgan.csv --reference_data_path data/real/train.csv
```
### TabSyn

```bash
python train_tabsyn.py --train_path data/real/train.csv --model_dir models/tabsyn
```

```bash
python generate_with_tabsyn.py --model_dir models/tabsyn --output_path data/synthetic/tabsyn.csv
```

### REaLTabFormer

```bash
python train_rtf.py --data_path data/real/train.csv --model_dir models/ --epochs 50 --batch_size 1024
```

```bash
python scripts/generate_with_rtf.py --model_path models/rtf_compute --output_path data/synthetic/rtf.csv --reference_data_path data/real/train.csv
```



## Evaluation

### Utility

```bash
python scripts/evaluate_utility.py --real_train_path data/real/train.csv --real_test_path data/real/test.csv --synthetic_paths data/synthetic/copula.csv  data/synthetic/ctgan.csv data/synthetic/tabsyn.csv data/synthetic/rtf.csv --synthetic_names GaussianCopula CTGAN TabSyn REaLTabFormer --output_dir results/utility --targets DEPARTURE_DELAY_MIN ARRIVAL_DELAY_MIN TURNAROUND_MIN --prediction_modes pre-tactical tactical
```

```bash
python scripts/plot_utility.py --results_dir results/utility --output_dir results/utility/plots
```

### Fidelity

```bash
python scripts/evaluate_fidelity.py --real_test_path data/real/test.csv --synthetic_paths data/synthetic/copula.csv  data/synthetic/ctgan.csv data/synthetic/tabsyn.csv data/synthetic/rtf.csv --synthetic_names GaussianCopula CTGAN TabSyn REaLTabFormer --metrics all --output_dir results/fidelity
```

```bash
python scripts/plot_fidelity.py --results_path results/fidelity/fidelity_results.csv --output_dir results/fidelity/plots --metrics all
```

### Privacy

```bash
python scripts/evaluate_privacy.py --real_train_path data/real/train.csv --real_validation_path data/real/test.csv --synthetic_paths data/synthetic/copula.csv  data/synthetic/ctgan.csv data/synthetic/tabsyn.csv data/synthetic/rtf.csv --synthetic_names GaussianCopula CTGAN TabSyn REaLTabFormer --output_dir results/privacy --metrics all --sample_size 1000 --num_iterations 3
```

```bash
python scripts/plot_privacy.py --results_dir results/privacy --output_dir results/privacy/plots --plot_types all
```