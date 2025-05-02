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
   - **Overall Utility** (scale 0-1): Represents how close the performance of a model trained on synthetic data is to one trained on real data. A value of 1.0 means synthetic data performs as well as real data.
   - **RMSE Utility**: Specific utility measure based on RMSE (closer to 1 is better)
   - **R² Utility**: Specific utility measure based on R² (closer to 1 is better)

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
*MAE values show similar patterns to RMSE but with smaller absolute values. The ranking of synthetic data generators remains consistent across metrics, with RTF generally performing best, while CTGAN , TVAE and Copula methods show higher error rates.*

![R²](utility/plots/departure_delay_min_pre-tactical/departure_delay_min_pre-tactical_r2.png)
*R² scores reveal that even with real data, departure delay prediction has moderate predictive power (typically between 0.1-0.3 with advanced models), reflecting the inherent unpredictability of flight delays using only pre-departure information. RTF synthetic data generally achieves R² values closest to real data. TVAE have negative R² values, indicating poor performance in capturing the variance in departure delays. Negative R² values suggest that the model performs worse than a simple mean prediction, indicating a lack of predictive power.*

### Utility Scores

![Utility Heatmap](utility/plots/departure_delay_min_pre-tactical/departure_delay_min_pre-tactical_utility_heatmap.png)
*This heatmap reveals how each synthetic dataset performs across different models relative to real data. RTF consistently shows the highest utility scores (darkest blue) across most models, particularly with ensemble methods. CTGAN performs well with certain models, while TVAE and Copula show lower utility overall.*

![Average Utility](utility/plots/departure_delay_min_pre-tactical/departure_delay_min_pre-tactical_avg_utility.png)
*Average utility scores provide a single metric for each synthetic dataset's quality. For departure delay prediction, RTF achieves the highest average utility (approximately 0.96), followed by CTGAN (around 0.68), with Copula  and TVAE showing significantly lower scores (below 0.6).*

### Feature Importance

![Average Feature Importance](utility/plots/departure_delay_min_pre-tactical/feature_importances/departure_delay_min_pre-tactical_all_models_feature_comparison.png)
*The average feature importance across all models for departure delay prediction shows that temporal features (Scheduled Hour, Day) and airport-specific features (Departure Airport, Arrival Airport) are the strongest predictors. RTF synthetic data most accurately preserves this pattern from real data, followed by CTGAN. TVAE and Copula show less alignment with real data feature importance rankings, explaining their lower utility scores. The feature importance analysis confirms that high-quality synthetic data captures the same predictive relationships as real data, which is crucial for operational forecasting.*

## Arrival Delay Prediction (Pre-tactical)

### Performance Metrics

![RMSE](utility/plots/arrival_delay_min_pre-tactical/arrival_delay_min_pre-tactical_rmse.png)
*RMSE for pre-tactical arrival delay prediction is typically higher than for departure delay, reflecting the additional uncertainty from en-route factors. The performance gap between real and synthetic data is slightly larger here, with RTF still performing best among synthetic generators.*

![MAE](utility/plots/arrival_delay_min_pre-tactical/arrival_delay_min_pre-tactical_mae.png)
*MAE values for arrival delay prediction show a similar pattern to RMSE. The relative ranking of synthetic data generators remains consistent, with RTF showing the lowest MAE among synthetic datasets.*

![R²](utility/plots/arrival_delay_min_pre-tactical/arrival_delay_min_pre-tactical_r2.png)
*R² scores for pre-tactical arrival delay prediction are generally lower than for departure delay, highlighting the increased difficulty of predicting arrival delays before the flight has departed. Even with real data, R² values rarely exceed 0.3 with the best models. TVAE shows negative R² values across all models, indicating poor utility in capturing the variance in arrival delays. This suggests that the model performs worse than a simple mean prediction, indicating a lack of predictive power.*

### Utility Scores

![Utility Heatmap](utility/plots/arrival_delay_min_pre-tactical/arrival_delay_min_pre-tactical_utility_heatmap.png)
*The utility heatmap for arrival delay prediction shows more variation across models than for departure delay. RTF still achieves the highest scores with ensemble methods. CTGAN performs competitively with certain models.*

![Average Utility](utility/plots/arrival_delay_min_pre-tactical/arrival_delay_min_pre-tactical_avg_utility.png)
*Average utility scores for arrival delay prediction show RTF leading with approximately 0.95, followed byCTGAN  around 0.56. Overall, the average utility scores are lower than for departure delay prediction, indicating that pre-tactical arrival delay prediction is more challenging.*

### Feature Importance

![Average Feature Importance](utility/plots/arrival_delay_min_pre-tactical/feature_importances/arrival_delay_min_pre-tactical_all_models_feature_comparison.png)
*Average feature importance across all models for arrival delay prediction reveals that Scheduled Hour, Scheduled Day Scheduled Duration, and airport-related features are the most significant predictors. The balance between temporal and spatial features is slightly different than for departure delay prediction, with duration playing a more prominent role. This makes sense, as scheduled duration is a key factor in determining arrival delay. RTF synthetic data most closely preserves this pattern from real data, while other generators show greater deviation in the relative importance of key features.*

## Arrival Delay Prediction (Tactical)

### Performance Metrics

![RMSE](utility/plots/arrival_delay_min_tactical/arrival_delay_min_tactical_rmse.png)
*Tactical arrival delay prediction shows substantially lower RMSE than pre-tactical prediction, with values typically 30-50% lower. This demonstrates the significant predictive value of including departure delay information.*

![MAE](utility/plots/arrival_delay_min_tactical/arrival_delay_min_tactical_mae.png)
*MAE values for tactical arrival delay prediction show similar improvements over pre-tactical mode.*

![R²](utility/plots/arrival_delay_min_tactical/arrival_delay_min_tactical_r2.png)
*R² scores in tactical mode are dramatically higher than in pre-tactical mode, often exceeding 0.8 with advanced models on real data. This substantial improvement confirms that knowing departure delay is critical for accurate arrival delay prediction. Synthetic datasets also show this pattern, with RTF, CTGAN and Copula achieving R² values above 0.7.*

### Utility Scores

![Utility Heatmap](utility/plots/arrival_delay_min_tactical/arrival_delay_min_tactical_utility_heatmap.png)
*The utility heatmap for tactical arrival delay prediction shows higher overall scores than pre-tactical mode. RTF and CTGAN both achieve excellent utility scores (above 0.85) with ensemble models.*

![Average Utility](utility/plots/arrival_delay_min_tactical/arrival_delay_min_tactical_avg_utility.png)
*Average utility scores in tactical mode are higher across all synthetic generators, with RTF reaching approximately 0.97 and CTGAN around 0.87. This indicates that the tactical relationship between departure and arrival delays is well-preserved in the synthetic datasets.*

### Feature Importance

![Average Feature Importance](utility/plots/arrival_delay_min_tactical/feature_importances/arrival_delay_min_tactical_all_models_feature_comparison.png)
*Average feature importance in tactical mode shows Departure Delay as the overwhelmingly dominant feature for predicting arrival delay, accounting for over 70% of the predictive power across all models. This dramatic shift from pre-tactical mode demonstrates how real-time departure information fundamentally changes the prediction task. All synthetic datasets correctly capture this strong dependency, though with slight variations in magnitude. RTF most accurately preserves the precise balance of feature importance seen in real data.*

## Turnaround Time Prediction (Pre-tactical)

### Performance Metrics

![RMSE](utility/plots/turnaround_min_pre-tactical/turnaround_min_pre-tactical_rmse.png)
*Turnaround time prediction shows higher RMSE values than delay prediction tasks, reflecting the greater variability in ground operations. The performance gap between real and synthetic data is more pronounced for this task.*

![MAE](utility/plots/turnaround_min_pre-tactical/turnaround_min_pre-tactical_mae.png)
*MAE values for turnaround prediction follow similar patterns to RMSE. The relative ranking of synthetic generators remains consistent across metrics, with RTF performing best among synthetic datasets, followed by CTGAN and Copula. TVAE shows the highest error rates, indicating poor performance in capturing turnaround time patterns.*

![R²](utility/plots/turnaround_min_pre-tactical/turnaround_min_pre-tactical_r2.png)
*R² scores for pre-tactical turnaround prediction are generally higher than for delay prediction, but still low overall (typically between 0.27-0.44). This indicates that turnaround time is more predictable than flight delays using only scheduled information. Although turnaround time has higher RMSE values, the R² scores suggest that the underlying relationships in the data are more consistent but the turnaround task varies more widely. RTF synthetic data achieves the highest R² scores among synthetic datasets, but still falls short of real data performance. TVAE shows negative R² values across all models, indicating poor utility in capturing the variance in turnaround times.*

### Utility Scores

![Utility Heatmap](utility/plots/turnaround_min_pre-tactical/turnaround_min_pre-tactical_utility_heatmap.png)
*The utility heatmap for turnaround prediction shows more variation across models than for delay prediction tasks. RTF achieves the highest scores with ensemble methods. CTGAN performs competitively with certain models, while TVAE and Copula show lower utility overall.*
![Average Utility](utility/plots/turnaround_min_pre-tactical/turnaround_min_pre-tactical_avg_utility.png)
*Average utility scores for turnaround prediction show RTF leading with approximately 0.97, followed by CTGAN around 0.74.*

### Feature Importance

![Average Feature Importance](utility/plots/turnaround_min_pre-tactical/feature_importances/turnaround_min_pre-tactical_all_models_feature_comparison.png)
*Average feature importance for turnaround time prediction shows that scheduled Hour, Duration, and Arrival Airport are the most significant predictors. The synthetic datasets show mixed success in preserving these relationships, with RTF performing best but still showing some discrepancies in the precise balance of feature importance compared to real data.*

## Turnaround Time Prediction (Tactical)

### Performance Metrics

![RMSE](utility/plots/turnaround_min_tactical/turnaround_min_tactical_rmse.png)
*Tactical turnaround prediction shows negligable improved RMSE compared to pre-tactical mode. This suggests actual flight information has less impact on turnaround times than on arrival delays. The performance gap between real and synthetic data persists in tactical mode.*

![MAE](utility/plots/turnaround_min_tactical/turnaround_min_tactical_mae.png)
*MAE values for tactical turnaround prediction show similar modest improvements over pre-tactical mode. All synthetic datasets maintain their relative ranking, with RTF performing best.*

![R²](utility/plots/turnaround_min_tactical/turnaround_min_tactical_r2.png)
*R² scores for tactical turnaround prediction does not change from pre-tactical mode, remaining low overall (typically between 0.27-0.44). This indicates that turnaround time is more predictable than flight delays using only scheduled information. RTF synthetic data achieves the highest R² scores among synthetic datasets, but still falls short of real data performance. TVAE shows negative R² values in most models, indicating poor utility in capturing the variance in turnaround times.*

### Utility Scores

![Utility Heatmap](utility/plots/turnaround_min_tactical/turnaround_min_tactical_utility_heatmap.png)
*The utility heatmap for tactical turnaround prediction does not change from pre-tactical mode. RTF achieves the highest scores with ensemble methods. CTGAN performs competitively with certain models, while TVAE and Copula show lower utility overall.*

![Average Utility](utility/plots/turnaround_min_tactical/turnaround_min_tactical_avg_utility.png)
*Average utility scores in tactical mode are similar to pre-tactical mode, with RTF leading with approximately 0.97 and CTGAN around 0.75.*

### Feature Importance

![Average Feature Importance](utility/plots/turnaround_min_tactical/feature_importances/turnaround_min_tactical_all_models_feature_comparison.png)
*Average feature importance in tactical mode shows that Scheduled Hour, Arrival Airport, and Duration are the most significant predictors. There is a slight shift in the relative importance of features compared to pre-tactical mode, with Arrival Airport becoming more significant. The new features introduced in tactical mode (e.g., Arrival Delay) do not significantly impact the prediction task. RTF synthetic data most closely preserves this pattern from real data, while other generators show greater deviation in the relative importance of key features.*

## Comparative Analysis

### Performance Across Prediction Tasks

When comparing performance across different prediction tasks:

1. **Tactical vs. Pre-tactical**: The impact of tactical information varies significantly by prediction task:
   - For arrival delay prediction, tactical mode dramatically improves performance, with R² values increasing from 0.3 to over 0.8 with real data and advanced models. This reflects the strong predictive relationship between departure and arrival delays.
   - For turnaround time prediction, tactical information provides negligible improvement, suggesting that real-time flight information has minimal impact on ground operations. R² scores remain in the 0.27-0.44 range in both modes.

2. **Delay vs. Turnaround**: Contrary to initial expectations, turnaround time prediction achieves higher R² scores (0.27-0.44) than departure and arrival delay prediction in pre-tactical mode (0.1-0.3). This suggests that while turnaround times have higher absolute error values (RMSE), they follow more consistent patterns that are easier for models to learn. Flight delays appear to be inherently more unpredictable when using only scheduled information.

3. **Model Complexity**: Tree-based ensemble models (Random Forest, XGBoost, CatBoost) consistently outperform the simpler Decision Tree model across all tasks and datasets. The performance gap is typically 20-40% in terms of error metrics, highlighting the complexity of the relationships in flight operations data.

### Synthetic Data Generator Comparison

The relative performance of different synthetic data generators is consistent across tasks:

1. **RTF (REaLTabFormer)**: Consistently achieves the highest utility scores across all prediction tasks, with average utilities ranging from 0.95 to 0.97. Its transformer-based architecture appears extremely effective at capturing the temporal dependencies and complex relationships in flight operations data, delivering synthetic data performance that comes remarkably close to real data.

2. **CTGAN**: Demonstrates solid performance as the second-best generator, typically achieving utility scores in the 0.56-0.87 range. It performs particularly well in tactical arrival delay prediction (0.87 utility) but shows more modest results in pre-tactical modes.

3. **Copula**: Shows moderate utility scores (typically 0.5-0.65) but performs consistently across different prediction tasks. Its performance is competitive with CTGAN in some scenarios, despite its simpler modeling approach.

4. **TVAE**: Generally shows the lowest utility scores and often produces negative R² values across multiple prediction tasks. This indicates significant difficulty in capturing the underlying patterns in the data, particularly for arrival delay and turnaround time prediction.

## Summary and Key Findings

The TSTR evaluation results provide several important insights:

1. **Synthetic Data Quality**: High-quality synthetic data (particularly RTF-generated) can achieve performance remarkably close to real data, with utility scores reaching 0.95-0.97 for most tasks. This demonstrates the tremendous potential of advanced synthetic data for flight operations modeling, offering a viable alternative to real data in scenarios where privacy or data access is constrained.

2. **Feature Preservation**: Feature importance analysis confirms that RTF synthetic data consistently preserves the same predictive relationships seen in real data. This faithful reproduction of feature importance patterns explains RTF's superior utility scores and validates its use for analytical and modeling purposes. CTGAN shows reasonable feature preservation, while TVAE and Copula demonstrate more significant deviations from real data patterns.

3. **Tactical Value**: The substantial performance improvement when moving from pre-tactical to tactical prediction for arrival delay (R² improvement from 0.3 to 0.8+) highlights the critical value of real-time departure information. This improvement is captured well by the better synthetic datasets, particularly RTF and CTGAN. In contrast, the negligible improvement in turnaround prediction suggests that ground operations follow different patterns that are less dependent on actual flight information.

4. **Task-Specific Performance**: Different prediction tasks demonstrate distinct characteristics:
   - Departure delay is moderately predictable (R² 0.1-0.3) using only scheduled information
   - Arrival delay is similarly challenging pre-tactically but becomes highly predictable (R² 0.8+) once departure information is available
   - Turnaround time shows more consistent patterns (R² 0.27-0.44) that remain stable regardless of whether real-time flight information is available

5. **Generator Ranking**: The consistent ranking of synthetic data generators (RTF > CTGAN > Copula > TVAE) across different prediction tasks suggests that certain architectural approaches are fundamentally better suited to capturing the complex relationships in flight operations data. RTF's transformer-based architecture appears particularly effective for this domain, while TVAE's variational autoencoder approach struggles to capture the relevant patterns.

These findings demonstrate the significant potential of synthetic flight operations data, particularly when generated using advanced methods like RTF. For practical applications, RTF-generated synthetic data can provide a highly viable alternative to real data, achieving performance that approaches 97% of real data utility across multiple prediction tasks. This makes it suitable for a wide range of applications, including model development, training, testing, and operational planning where privacy concerns or data access limitations would otherwise be constraining factors.