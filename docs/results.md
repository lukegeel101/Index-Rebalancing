# Reported model results

These values are preserved from the original exploratory notebook and report.
CI verifies the committed dataset but does not retrain the models or independently reproduce these notebook-era model metrics.

## Summary

| Experiment | Reported result |
| --- | --- |
| Linear regression | Average score of -0.1 in the original run |
| Linear threshold screen | 8.24% average actual change among predictions above 7% |
| Initial LSTM | Reported mean absolute error of 0.194 |
| Five-fold LSTM run | Reported error of 0.132 in the final plotted fold |

The original narrative used both `MAE` and `MSE` for the value `0.132`.
Because the notebook output needed to disambiguate that label is not committed, this release describes it conservatively as a reported error rather than upgrading the claim.

## Linear-regression threshold observations

| Actual change | Predicted change |
| ---: | ---: |
| 8.139869 | 7.252531 |
| 6.788077 | 8.755144 |
| 20.017240 | 8.771344 |
| 5.838512 | 8.278123 |
| 13.982063 | 7.147442 |
| 14.456949 | 8.085960 |
| 8.574881 | 8.280803 |
| 9.626756 | 8.463116 |
| 0.129809 | 8.061031 |
| -3.988344 | 8.188798 |
| 7.583768 | 7.124308 |
| 5.870947 | 8.966869 |
| 12.333095 | 7.076790 |
| 11.145196 | 9.175817 |
| 20.074626 | 7.959058 |
| 14.506850 | 8.513511 |
| 3.864210 | 7.944864 |
| 11.216861 | 8.817162 |
| 17.317569 | 9.026893 |
| 7.387385 | 11.510598 |
| 8.369286 | 8.518804 |
| 13.048085 | 8.883695 |
| 2.650797 | 8.137456 |
| 11.066149 | 9.716412 |
| 6.897417 | 10.332827 |
| -3.012613 | 8.042343 |
| 6.066175 | 7.767262 |
| -1.764384 | 8.110319 |
| 1.555865 | 7.585922 |
| 10.234158 | 9.394775 |
| 0.833764 | 8.469924 |
| 13.903845 | 8.153528 |
| 7.182159 | 7.932592 |
