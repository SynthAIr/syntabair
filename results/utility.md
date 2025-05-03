# Train on Synthetic, Test on Real (TSTR) Evaluation Results

## Table of Contents
- [Train on Synthetic, Test on Real (TSTR) Evaluation Results](#train-on-synthetic-test-on-real-tstr-evaluation-results)
  - [Table of Contents](#table-of-contents)
  - [Overview of Result Types](#overview-of-result-types)
  - [Prediction Targets and Modes](#prediction-targets-and-modes)
  - [Departure Delay Prediction (Pre-tactical)](#departure-delay-prediction-pre-tactical)
    - [Performance Metrics](#performance-metrics)
    - [Utility Scores](#utility-scores)
    - [Feature Importance](#feature-importance)
  - [Arrival Delay Prediction (Pre-tactical)](#arrival-delay-prediction-pre-tactical)
    - [Performance Metrics](#performance-metrics-1)
    - [Utility Scores](#utility-scores-1)
    - [Feature Importance](#feature-importance-1)
  - [Arrival Delay Prediction (Tactical)](#arrival-delay-prediction-tactical)
    - [Performance Metrics](#performance-metrics-2)
    - [Utility Scores](#utility-scores-2)
    - [Feature Importance](#feature-importance-2)
  - [Turnaround Time Prediction (Pre-tactical)](#turnaround-time-prediction-pre-tactical)
    - [Performance Metrics](#performance-metrics-3)
    - [Utility Scores](#utility-scores-3)
    - [Feature Importance](#feature-importance-3)
  - [Turnaround Time Prediction (Tactical)](#turnaround-time-prediction-tactical)
    - [Performance Metrics](#performance-metrics-4)
    - [Utility Scores](#utility-scores-4)
    - [Feature Importance](#feature-importance-4)
  - [Comparative Analysis](#comparative-analysis)
    - [Performance Across Prediction Tasks](#performance-across-prediction-tasks)
    - [Synthetic Data Generator Comparison](#synthetic-data-generator-comparison)
  - [Summary and Key Findings](#summary-and-key-findings)

## Overview of Result Types

The evaluation produces three main types of results:

1. **Performance Metrics**: Quantitative measures of prediction accuracy
   - **Root Mean Squared Error (RMSE)**: Measures the square root of the average squared differences between predicted and actual values. Lower values indicate better performance. RMSE penalizes large errors more heavily.
   - **Mean Absolute Error (MAE)**: Measures the average absolute differences between predicted and actual values. Lower values indicate better performance. MAE is more robust to outliers than RMSE.
   - **R² Score**: Measures the proportion of variance in the dependent variable that is predictable from the independent variables. Values range from 0 to 1, with higher values indicating better fit.

2. **Utility Scores**: Comparative measures of synthetic data quality
   - **Utility Heatmap**: Visualizes how well each synthetic dataset performs across different models relative to real data. Darker colors indicate better performance.
   - **Average Utility**: Provides a single metric for each synthetic dataset's quality. Higher values indicate better performance. This score is calculated as the average of the utility scores across all models for each dataset.

3. **Feature Importance**: Insights into which variables drive predictions
   - Shows which features have the most predictive power across all models
   - Helps validate if synthetic data preserves the same feature relationships as real data

## Prediction Targets and Modes

The evaluation covers three prediction targets, each important for flight operations:

- **Departure Delay**: Predicting how many minutes a flight will depart later than scheduled
- **Arrival Delay**: Predicting how many minutes a flight will arrive later than scheduled
- **Turnaround Time**: Predicting how long an aircraft needs to prepare for its next flight

Each target is evaluated in one or two prediction modes:

- **Pre-tactical**: Predictions made before departure using only scheduled information (planning phase)
- **Tactical**: Predictions made with real-time information during operations (execution phase)

## Departure Delay Prediction (Pre-tactical)

### Performance Metrics

![RMSE](utility/plots/departure_delay_min_pre-tactical/departure_delay_min_pre-tactical_rmse.png)
*RMSE comparison across models and datasets. Lower values indicate better performance. Real data yields the best performance, with RTF-generated synthetic data showing the closest performance to real data across most models.*

![MAE](utility/plots/departure_delay_min_pre-tactical/departure_delay_min_pre-tactical_mae.png)
*MAE values show similar patterns to RMSE but with smaller absolute values. The ranking of synthetic data generators remains consistent across metrics, with RTF generally performing best*

![R²](utility/plots/departure_delay_min_pre-tactical/departure_delay_min_pre-tactical_r2.png)
*R² scores reveal that even with real data, departure delay prediction has moderate predictive power (typically between 0.1-0.3 with advanced models), reflecting the inherent unpredictability of flight delays using only pre-departure information. RTF synthetic data generally achieves R² values closest to real data.*

### Utility Scores

![Utility Heatmap](utility/plots/departure_delay_min_pre-tactical/departure_delay_min_pre-tactical_utility_heatmap.png)
*This heatmap reveals how each synthetic dataset performs across different models relative to real data. RTF consistently shows the highest utility scores (darkest blue) across most models, particularly with ensemble methods. TabSyn and CTGAN perform well with certain models, while Copula shows lower utility overall.*

![Average Utility](utility/plots/departure_delay_min_pre-tactical/departure_delay_min_pre-tactical_avg_utility.png)
*Average utility scores provide a single metric for each synthetic dataset's quality. For departure delay prediction, RTF achieves the highest average utility (approximately 0.96), followed by Tabsyn (around 0.76)  and CTGAN (around 0.68), with Copula showing significantly lower scores (below 0.6).*

### Feature Importance

![Average Feature Importance](utility/plots/departure_delay_min_pre-tactical/feature_importances/departure_delay_min_pre-tactical_all_models_feature_comparison.png)
*The average feature importance across all models for departure delay prediction shows that temporal features (Scheduled Hour) and airport-specific features (Departure Airport, Arrival Airport) are the strongest predictors. RTF synthetic data most accurately preserves this pattern from real data, followed by TabSyn and CTGAN.  Copula shows less alignment with real data feature importance rankings, explaining its lower utility scores. The feature importance analysis confirms that high-quality synthetic data captures the same predictive relationships as real data, which is crucial for operational forecasting.*

## Arrival Delay Prediction (Pre-tactical)

### Performance Metrics

![RMSE](utility/plots/arrival_delay_min_pre-tactical/arrival_delay_min_pre-tactical_rmse.png)
*RMSE for pre-tactical arrival delay prediction is typically higher than for departure delay, reflecting the additional uncertainty from en-route factors. The performance gap between real and synthetic data is slightly larger here, with RTF still performing best among synthetic generators.*

![MAE](utility/plots/arrival_delay_min_pre-tactical/arrival_delay_min_pre-tactical_mae.png)
*MAE values for arrival delay prediction show a similar pattern to RMSE. The relative ranking of synthetic data generators remains consistent, with RTF showing the lowest MAE among synthetic datasets.*

![R²](utility/plots/arrival_delay_min_pre-tactical/arrival_delay_min_pre-tactical_r2.png)
*R² scores for pre-tactical arrival delay prediction are generally lower than for departure delay, highlighting the increased difficulty of predicting arrival delays before the flight has departed. Even with real data, R² values rarely exceed 0.3 with the best models.*

### Utility Scores

![Utility Heatmap](utility/plots/arrival_delay_min_pre-tactical/arrival_delay_min_pre-tactical_utility_heatmap.png)
*The utility heatmap for arrival delay prediction shows more variation across models than for departure delay. RTF still achieves the highest scores with ensemble methods. TabSyn performs competitively with certain models.*

![Average Utility](utility/plots/arrival_delay_min_pre-tactical/arrival_delay_min_pre-tactical_avg_utility.png)
*Average utility scores for arrival delay prediction show RTF leading with approximately 0.95, followed by TabSyn  around 0.64. Overall, the average utility scores are lower than for departure delay prediction, indicating that pre-tactical arrival delay prediction is more challenging.*

### Feature Importance

![Average Feature Importance](utility/plots/arrival_delay_min_pre-tactical/feature_importances/arrival_delay_min_pre-tactical_all_models_feature_comparison.png)
*Average feature importance across all models for arrival delay prediction reveals that Scheduled Hour, Duration, Day and airport-related features are the most significant predictors. The balance between temporal and spatial features is slightly different than for departure delay prediction, with duration playing a more prominent role. This makes sense, as scheduled duration is a key factor in determining arrival delay. RTF synthetic data most closely preserves this pattern from real data, while other generators show greater deviation in the relative importance of key features.*

## Arrival Delay Prediction (Tactical)

### Performance Metrics

![RMSE](utility/plots/arrival_delay_min_tactical/arrival_delay_min_tactical_rmse.png)
*Tactical arrival delay prediction shows substantially lower RMSE than pre-tactical prediction, with values typically 30-50% lower. This demonstrates the significant predictive value of including departure delay information.*

![MAE](utility/plots/arrival_delay_min_tactical/arrival_delay_min_tactical_mae.png)
*MAE values for tactical arrival delay prediction show similar improvements over pre-tactical mode.*

![R²](utility/plots/arrival_delay_min_tactical/arrival_delay_min_tactical_r2.png)
*R² scores in tactical mode are dramatically higher than in pre-tactical mode, often exceeding 0.8 with some models on real data. This substantial improvement confirms that knowing departure delay is critical for accurate arrival delay prediction. Synthetic datasets also show this pattern, with RTF, TabSyn and CTGAN achieving R² values above 0.7.*

### Utility Scores

![Utility Heatmap](utility/plots/arrival_delay_min_tactical/arrival_delay_min_tactical_utility_heatmap.png)
*The utility heatmap for tactical arrival delay prediction shows higher overall scores than pre-tactical mode. RTF and TabSyn both achieve high utility scores (above 0.9) with ensemble models.*

![Average Utility](utility/plots/arrival_delay_min_tactical/arrival_delay_min_tactical_avg_utility.png)
*Average utility scores in tactical mode are higher across all synthetic generators, with RTF reaching approximately 0.97 and TabSyn around 0.9. This indicates that the tactical relationship between departure and arrival delays is well-preserved in the synthetic datasets.*

### Feature Importance

![Average Feature Importance](utility/plots/arrival_delay_min_tactical/feature_importances/arrival_delay_min_tactical_all_models_feature_comparison.png)
*Average feature importance in tactical mode shows Departure Delay as the overwhelmingly dominant feature for predicting arrival delay, accounting for over 90% of the predictive power across all models. This dramatic shift from pre-tactical mode demonstrates how real-time departure information fundamentally changes the prediction task. All synthetic datasets correctly capture this strong dependency, though with slight variations in magnitude.*

## Turnaround Time Prediction (Pre-tactical)

### Performance Metrics

![RMSE](utility/plots/turnaround_min_pre-tactical/turnaround_min_pre-tactical_rmse.png)
*Turnaround time prediction shows higher RMSE values than delay prediction tasks, reflecting the greater variability in ground operations.*

![MAE](utility/plots/turnaround_min_pre-tactical/turnaround_min_pre-tactical_mae.png)
*MAE values for turnaround prediction follow similar patterns to RMSE. The relative ranking of synthetic generators remains consistent across metrics, with RTF and TabSyn performing best among synthetic datasets.*

![R²](utility/plots/turnaround_min_pre-tactical/turnaround_min_pre-tactical_r2.png)
*R² scores for pre-tactical turnaround prediction are generally higher than for delay prediction, but still low overall (typically between 0.27-0.44). This indicates that turnaround time is more predictable than flight delays using only scheduled information. Although turnaround time has higher RMSE values, the R² scores suggest that the underlying relationships in the data are more consistent but the turnaround task varies more widely. RTF synthetic data achieves the highest R² scores among synthetic datasets.*

### Utility Scores

![Utility Heatmap](utility/plots/turnaround_min_pre-tactical/turnaround_min_pre-tactical_utility_heatmap.png)
*RTF achieves the highest scores with ensemble methods. TabSyn performs competitively as well, while CTGAN and Copula show lower utility overall.*
![Average Utility](utility/plots/turnaround_min_pre-tactical/turnaround_min_pre-tactical_avg_utility.png)
*Average utility scores for turnaround prediction show RTF leading with approximately 0.97, followed by TabSyn around 0.93.*

### Feature Importance

![Average Feature Importance](utility/plots/turnaround_min_pre-tactical/feature_importances/turnaround_min_pre-tactical_all_models_feature_comparison.png)
*Average feature importance for turnaround time prediction shows that scheduled Hour, Duration, and Arrival Airport (related toground handling efficiency) are the most significant predictors. The synthetic datasets show mixed success in preserving these relationships, with RTF and TabSyn performing best but still showing some discrepancies in the precise balance of feature importance compared to real data.*

## Turnaround Time Prediction (Tactical)

### Performance Metrics

![RMSE](utility/plots/turnaround_min_tactical/turnaround_min_tactical_rmse.png)
*Tactical turnaround prediction shows negligable improved RMSE compared to pre-tactical mode. This suggests actual flight information has less impact on turnaround times than on arrival delays. The performance gap between real and synthetic data persists in tactical mode.*

![MAE](utility/plots/turnaround_min_tactical/turnaround_min_tactical_mae.png)
*MAE values for tactical turnaround prediction show similar modest improvements over pre-tactical mode. All synthetic datasets maintain their relative ranking, with RTF and TabSyn performing best.*

![R²](utility/plots/turnaround_min_tactical/turnaround_min_tactical_r2.png)
*R² scores for tactical turnaround prediction does not change from pre-tactical mode, remaining low overall (typically between 0.27-0.44). This indicates that turnaround time is more predictable than flight delays using only scheduled information. RTF synthetic data achieves the highest R² scores among synthetic datasets. GaussianCopula shows really low R² values in most models, indicating poor utility in capturing the variance in turnaround times.*

### Utility Scores

![Utility Heatmap](utility/plots/turnaround_min_tactical/turnaround_min_tactical_utility_heatmap.png)
*The utility heatmap for tactical turnaround prediction does not change from pre-tactical mode. RTF and TabSyn achieve the highest scores with ensemble methods. CTGAN performs competitively with certain models, while Copula shows lower utility overall.*

![Average Utility](utility/plots/turnaround_min_tactical/turnaround_min_tactical_avg_utility.png)
*Average utility scores in tactical mode are similar to pre-tactical mode, with RTF leading with approximately 0.97 and TabSyn around 0.93.*

### Feature Importance

![Average Feature Importance](utility/plots/turnaround_min_tactical/feature_importances/turnaround_min_tactical_all_models_feature_comparison.png)
*Average feature importance in tactical mode shows that Scheduled Hour, Arrival Airport, and Duration are the most significant predictors. There is a slight shift in the relative importance of features compared to pre-tactical mode, with Arrival Airport becoming more significant. The new features introduced in tactical mode (e.g., Arrival Delay) do not significantly impact the prediction task. RTF and TabSyn synthetic data most closely preserves this pattern from real data, while other generators show greater deviation in the relative importance of key features.*
