
## Data Preparation

The original dataset contains the following columns:

```python
['STATUS_KEY', 'IATA_CARRIER_CODE', 'ICAO_CARRIER_CODE', 'FLIGHT_NUMBER',
'FLIGHT_TYPE', 'FLIGHT_STATE', 'DEPARTURE_IATA_AIRPORT_CODE',
'DEPARTURE_ICAO_AIRPORT_CODE', 'DEPARTURE_FAA_AIRPORT_CODE',
'DEPARTURE_COUNTRY_CODE', 'DEPARTURE_TERMINAL', 'DEPARTURE_GATE',
'CHECK_IN_COUNTER', 'SCHEDULED_DEPARTURE_TIME_LOCAL',
'SCHEDULED_DEPARTURE_DATE_LOCAL',
'DEPARTURE_ESTIMATED_OUTGATE_TIMELINESS',
'DEPARTURE_ESTIMATED_OUTGATE_VARIATION',
'DEPARTURE_ESTIMATED_OUTGATE_UTC', 'DEPARTURE_ESTIMATED_OUTGATE_LOCAL',
'DEPARTURE_ESTIMATED_OFFGROUND_UTC',
'DEPARTURE_ESTIMATED_OFFGROUND_LOCAL',
'DEPARTURE_ACTUAL_OUTGATE_TIMELINESS',
'DEPARTURE_ACTUAL_OUTGATE_VARIATION', 'DEPARTURE_ACTUAL_OUTGATE_UTC',
'DEPARTURE_ACTUAL_OUTGATE_LOCAL', 'DEPARTURE_ACTUAL_OFFGROUND_UTC',
'DEPARTURE_ACTUAL_OFFGROUND_LOCAL', 'ARRIVAL_IATA_AIRPORT_CODE',
'ARRIVAL_ICAO_AIRPORT_CODE', 'ARRIVAL_FAA_AIRPORT_CODE',
'ARRIVAL_COUNTRY_CODE', 'ARRIVAL_TERMINAL', 'ARRIVAL_GATE', 'BAGGAGE',
'SCHEDULED_ARRIVAL_TIME_LOCAL', 'ARRIVAL_ESTIMATED_INGATE_TIMELINESS',
'ARRIVAL_ESTIMATED_INGATE_VARIATION', 'ARRIVAL_ESTIMATED_ONGROUND_UTC',
'ARRIVAL_ESTIMATED_ONGROUND_LOCAL', 'ARRIVAL_ESTIMATED_INGATE_UTC',
'ARRIVAL_ESTIMATED_INGATE_LOCAL', 'ARRIVAL_ACTUAL_INGATE_TIMELINESS',
'ARRIVAL_ACTUAL_INGATE_VARIATION', 'ARRIVAL_ACTUAL_INGATE_UTC',
'ARRIVAL_ACTUAL_INGATE_LOCAL', 'ARRIVAL_ACTUAL_ONGROUND_UTC',
'ARRIVAL_ACTUAL_ONGROUND_LOCAL', 'AIRCRAFT_TYPE_IATA',
'AIRCRAFT_TYPE_ICAO', 'AIRCRAFT_REGISTRATION_NUMBER',
'GENERAL_AVIATION_FLIGHT_IDENTIFIER', 'SERVICE_TYPE',
'DIVERSION_IATA_AIRPORT_CODE', 'DIVERSION_ICAO_AIRPORT_CODE',
'DIVERSION_FAA_AIRPORT_CODE', 'SCHEDULE_INSTANCE_KEY',
'OPERATING_SCHEDULE_INSTANCE_KEY', 'IS_OPERATING_CARRIER',
'ACTUAL_FIRST_CLASS_SEATS', 'ACTUAL_BUSINESS_CLASS_SEATS',
'ACTUAL_PREMIUM_ECONOMY_CLASS_SEATS', 'ACTUAL_ECONOMY_PLUS_CLASS_SEATS',
'ACTUAL_ECONOMY_CLASS_SEATS', 'ACTUAL_TOTAL_SEATS',
'PREDICTED_FIRST_CLASS_SEATS', 'PREDICTED_BUSINESS_CLASS_SEATS',
'PREDICTED_PREMIUM_ECONOMY_CLASS_SEATS',
'PREDICTED_ECONOMY_PLUS_CLASS_SEATS', 'PREDICTED_ECONOMY_CLASS_SEATS',
'PREDICTED_TOTAL_SEATS', 'ORIGIN_MESSAGE_ID',
'ORIGIN_MESSAGE_TIMESTAMP']
```

To prepare the data, run the following script:

```bash
python prepare_data.py --input_path data/flights.csv --output_train_path data/real/train.csv --output_test_path data/real/test.csv --test_size 0.2
```


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


## Acknowledgments

This project incorporates code from the following open-source projects:

- [TabSyn](https://github.com/amazon-science/tabsyn) by Amazon Science, licensed under the Apache License 2.0.
- [REaLTabFormer](https://github.com/worldbank/REaLTabFormer) by The World Bank, licensed under the MIT License.

Modifications have been made to the original code to suit this project's requirements.
