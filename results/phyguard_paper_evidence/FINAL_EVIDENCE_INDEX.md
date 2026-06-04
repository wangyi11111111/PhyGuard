# PhyGuard paper evidence index

All paper-facing tables use display names: MagiNet, SAITS, BRITS, and ImputeFormer. Configuration details are described in the experimental setup text rather than encoded in table names.

## Generality by backbone

    backbone  base_mae_mean  phyguard_mae_mean  gain_pct_mean  gain_pct_std  wins  runs  gate_mean  delta_abs_mean  failure_score_mean  win_rate_pct
       BRITS       0.386274           0.356274       8.491951      3.955298    45    45   0.702097        0.107265             0.31175    100.000000
ImputeFormer       0.398566           0.370704       7.202371      3.354640    45    45   0.671381        0.090107             0.31175    100.000000
     MagiNet       0.361041           0.346078       5.219975      3.630740    43    45   0.610572        0.068595             0.31175     95.555556
       SAITS       0.378871           0.353655       7.254002      4.136444    45    45   0.692380        0.095285             0.31175    100.000000

## Scenario gains

             scenario  base_mae_mean  phyguard_mae_mean  gain_pct_mean  gain_pct_std  wins  runs  gate_mean  delta_abs_mean  failure_score_mean  win_rate_pct
incident_perturbation       0.284450           0.261088       7.919462      3.268783    60    60   0.625325        0.075998            0.076351    100.000000
    random_missing_50       0.265964           0.241651       8.976516      3.842290    60    60   0.630871        0.073027            0.076351    100.000000
    sensor_failure_30       0.593150           0.567295       4.230247      2.960246    58    60   0.751127        0.121915            0.782550     96.666667

## Missing-rate robustness

 missing_rate  base_mae_mean  phyguard_mae_mean  gain_pct_mean  gain_pct_std  wins  runs  gate_mean  delta_abs_mean  failure_score_mean  win_rate_pct
           30       0.291076           0.263921       8.744291      4.935424    18    18   0.653771        0.061198            0.044705         100.0
           50       0.315805           0.291209       7.218738      4.153053    18    18   0.640119        0.065988            0.072450         100.0
           70       0.340745           0.320773       5.902405      1.945440    18    18   0.622420        0.069580            0.093638         100.0

## Ablation

                variant  masked_mae_mean  masked_mae_std  gain_pct_mean  gain_pct_std  wins  runs  gate_mean  delta_abs_mean  win_rate_pct  phyguard_gain_vs_variant_pct
failure_aware_soft_harm         0.363818        0.080193       6.975529      4.667137    18    18   0.704419        0.092895    100.000000                 0.000000
         soft_harm_0.05         0.364181        0.082160       6.969617      4.802677    18    18   0.701977        0.096859    100.000000                 0.099749
         soft_harm_0.20         0.368315        0.075934       5.541538      4.667086    18    18   0.712583        0.078644    100.000000                 1.221059
                   full         0.381949        0.075647       2.095603      1.776349    18    18   0.702174        0.033523    100.000000                 4.746905
       no_failure_score         0.381978        0.075728       2.091405      1.743867    18    18   0.700658        0.033196    100.000000                 4.754260
                no_gate         0.381984        0.075705       2.095522      1.670260    18    18   1.000000        0.028787    100.000000                 4.755625
    no_physics_residual         0.382868        0.076167       1.880269      1.712874    17    18   0.697929        0.034468     94.444444                 4.975684
   no_temporal_features         0.386407        0.076805       1.001588      1.615849    14    18   0.600940        0.028350     77.777778                 5.845860

## Failure-aware harm vs fixed harm

             scenario  old_fixed_harm_gain  phyguard_gain  delta_gain
incident_perturbation             0.494481  7.919462    7.424981
    random_missing_50             0.802173  8.976516    8.174342
    sensor_failure_30             2.774726  4.230247    1.455521

## Visual case

See `../phyguard_visual_case/figure_phyguard_case.png`.
