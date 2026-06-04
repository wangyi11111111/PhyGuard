# Missing-rate robustness

Protocol: PEMS08, PEMS-BAY, METR-LA x random_missing_30/50/70 x MagiNet/SAITS x 3 seeds.

## By missing rate

 missing_rate  base_mae_mean  phyguard_mae_mean  gain_pct_mean  gain_pct_std  wins  runs  gate_mean  delta_abs_mean  failure_score_mean  win_rate_pct
           30       0.291076           0.263921       8.744291      4.935424    18    18   0.653771        0.061198            0.044705         100.0
           50       0.315805           0.291209       7.218738      4.153053    18    18   0.640119        0.065988            0.072450         100.0
           70       0.340745           0.320773       5.902405      1.945440    18    18   0.622420        0.069580            0.093638         100.0

## By dataset and missing rate

 dataset  missing_rate  base_mae_mean  phyguard_mae_mean  gain_pct_mean  gain_pct_std  wins  runs  win_rate_pct
 METR-LA            30       0.355331           0.336147       5.343152      1.271993     6     6         100.0
 METR-LA            50       0.407536           0.382638       6.100235      0.497377     6     6         100.0
 METR-LA            70       0.440033           0.421853       4.116678      0.284944     6     6         100.0
PEMS-BAY            30       0.314554           0.266989      14.631585      2.392824     6     6         100.0
PEMS-BAY            50       0.331165           0.289061      12.522907      1.068001     6     6         100.0
PEMS-BAY            70       0.356840           0.327909       7.909933      1.042773     6     6         100.0
  PEMS08            30       0.203343           0.188628       6.258137      3.547436     6     6         100.0
  PEMS08            50       0.208715           0.201927       3.033071      0.984171     6     6         100.0
  PEMS08            70       0.225362           0.212556       5.680605      1.725604     6     6         100.0

## Significance

               group  n  mean_abs_improvement  mean_gain_pct  paired_t      p_value
overall_missing_rate 54              0.023908       7.288478 10.891210 3.907166e-15
          missing_30 18              0.027155       8.744291  5.766701 2.278998e-05
          missing_50 18              0.024597       7.218738  6.086027 1.210965e-05
          missing_70 18              0.019973       5.902405  8.928876 7.930302e-08
