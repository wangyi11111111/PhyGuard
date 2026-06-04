# Missing-rate robustness summary

Win-rate columns are intentionally omitted.

## By missing rate

 missing_rate  base_mae_mean  phyguard_mae_mean  gain_pct_mean  gain_pct_std  gate_mean  delta_abs_mean  failure_score_mean
           30       0.291076           0.263921       8.744291      4.935424   0.653771        0.061198            0.044705
           50       0.315805           0.291209       7.218738      4.153053   0.640119        0.065988            0.072450
           70       0.340745           0.320773       5.902405      1.945440   0.622420        0.069580            0.093638

## By dataset and missing rate

 dataset  missing_rate  base_mae_mean  phyguard_mae_mean  gain_pct_mean  gain_pct_std
 METR-LA            30       0.355331           0.336147       5.343152      1.271993
 METR-LA            50       0.407536           0.382638       6.100235      0.497377
 METR-LA            70       0.440033           0.421853       4.116678      0.284944
PEMS-BAY            30       0.314554           0.266989      14.631585      2.392824
PEMS-BAY            50       0.331165           0.289061      12.522907      1.068001
PEMS-BAY            70       0.356840           0.327909       7.909933      1.042773
  PEMS08            30       0.203343           0.188628       6.258137      3.547436
  PEMS08            50       0.208715           0.201927       3.033071      0.984171
  PEMS08            70       0.225362           0.212556       5.680605      1.725604

## By backbone and missing rate

backbone  missing_rate  base_mae_mean  phyguard_mae_mean  gain_pct_mean  gain_pct_std
 MagiNet            30       0.222707           0.206419       7.091382      4.628265
 MagiNet            50       0.245421           0.227571       6.745627      4.137457
 MagiNet            70       0.284388           0.269486       5.517302      1.709074
   SAITS            30       0.359446           0.321424      10.397200      4.918693
   SAITS            50       0.386189           0.354847       7.691849      4.362282
   SAITS            70       0.397103           0.372059       6.287509      2.188137

