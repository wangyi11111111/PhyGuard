# PhyGuard Paper Evidence Tables

This folder is derived from the finalized paired experiments. Win-rate columns are intentionally omitted; paper-facing summaries report MAE, mean reduction, standard deviation, significance, and complexity.

## Generality

    backbone  base_mae_mean  phyguard_mae_mean  gain_pct_mean  gain_pct_std  gate_mean  delta_abs_mean  failure_score_mean
       BRITS       0.386274           0.356274       8.491951      3.955298   0.702097        0.107265             0.31175
ImputeFormer       0.398566           0.370704       7.202371      3.354640   0.671381        0.090107             0.31175
     MagiNet       0.361041           0.346078       5.219975      3.630740   0.610572        0.068595             0.31175
       SAITS       0.378871           0.353655       7.254002      4.136444   0.692380        0.095285             0.31175

## Explainability by scenario

             scenario  base_mae_mean  phyguard_mae_mean  gain_pct_mean  gain_pct_std  gate_mean  delta_abs_mean  failure_score_mean
incident_perturbation       0.284450           0.261088       7.919462      3.268783   0.625325        0.075998            0.076351
    random_missing_50       0.265964           0.241651       8.976516      3.842290   0.630871        0.073027            0.076351
    sensor_failure_30       0.593150           0.567295       4.230247      2.960246   0.751127        0.121915            0.782550

## Best baseline vs best PhyGuard variant

 dataset              scenario best_base_model  best_base_mae_mean  best_base_mae_std       best_plugin_model  best_plugin_mae_mean  best_plugin_mae_std  best_gain_pct
 METR-LA incident_perturbation    ImputeFormer            0.367067           0.035933 ImputeFormer + PhyGuard              0.345526             0.029908       5.868639
 METR-LA     random_missing_50    ImputeFormer            0.337071           0.024463 ImputeFormer + PhyGuard              0.318085             0.022328       5.632664
 METR-LA     sensor_failure_30           SAITS            0.501350           0.060989        SAITS + PhyGuard              0.490817             0.054119       2.101006
PEMS-BAY incident_perturbation         MagiNet            0.263218           0.001748      MagiNet + PhyGuard              0.237289             0.004663       9.850985
PEMS-BAY     random_missing_50         MagiNet            0.245374           0.001261      MagiNet + PhyGuard              0.216421             0.002990      11.799353
PEMS-BAY     sensor_failure_30           SAITS            0.531497           0.024573        SAITS + PhyGuard              0.514987             0.023243       3.106439
  PEMS03 incident_perturbation         MagiNet            0.173436           0.004439      MagiNet + PhyGuard              0.159313             0.003943       8.143407
  PEMS03     random_missing_50         MagiNet            0.150137           0.001479      MagiNet + PhyGuard              0.134583             0.000813      10.360188
  PEMS03     sensor_failure_30           SAITS            0.497538           0.053898        SAITS + PhyGuard              0.483827             0.051848       2.755808
  PEMS04 incident_perturbation         MagiNet            0.172308           0.001515      MagiNet + PhyGuard              0.163441             0.002261       5.146554
  PEMS04     random_missing_50         MagiNet            0.156573           0.001815      MagiNet + PhyGuard              0.145805             0.001258       6.877244
  PEMS04     sensor_failure_30           SAITS            0.414956           0.069905        SAITS + PhyGuard              0.387335             0.067899       6.656292
  PEMS08 incident_perturbation         MagiNet            0.149354           0.004172      MagiNet + PhyGuard              0.146787             0.004996       1.718751
  PEMS08     random_missing_50         MagiNet            0.133286           0.001819      MagiNet + PhyGuard              0.129994             0.001669       2.469964
  PEMS08     sensor_failure_30           BRITS            0.425740           0.008261        BRITS + PhyGuard              0.414781             0.010029       2.574198

## Complexity

 dataset  nodes  feature_dim  hidden_dim  extra_trainable_params benchmark_device  extra_forward_ms_per_batch batch_shape
  PEMS03    358           12          64                    5954             cuda                    0.846305 16x12x358x1
  PEMS04    307           12          64                    5954             cuda                    0.601794 16x12x307x1
  PEMS08    170           12          64                    5954             cuda                    0.334387 16x12x170x1
PEMS-BAY    325           12          64                    5954             cuda                    0.641956 16x12x325x1
 METR-LA    207           12          64                    5954             cuda                    0.386161 16x12x207x1

## Significance

                         group   n  mean_abs_improvement  mean_gain_pct  paired_t      p_value
                       overall 180              0.024510       7.042075 20.186651 5.287264e-48
scenario:incident_perturbation  60              0.023361       7.919462 13.719377 5.197308e-20
    scenario:random_missing_50  60              0.024313       8.976516 13.209054 2.833789e-19
    scenario:sensor_failure_30  60              0.025855       4.230247  9.719334 7.387907e-14
                backbone:BRITS  45              0.029999       8.491951 14.931962 7.745137e-19
         backbone:ImputeFormer  45              0.027861       7.202371  8.818121 2.785106e-11
              backbone:MagiNet  45              0.014963       5.219975  9.262042 6.728534e-12
                backbone:SAITS  45              0.025216       7.254002 12.107983 1.333645e-15

