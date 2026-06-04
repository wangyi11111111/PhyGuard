# Final Evidence Index

The evidence package intentionally omits win-rate statistics. Use MAE, mean reduction, standard deviation, paired significance, and complexity tables for paper reporting.

- `ablation_overall_display.csv`: 8 rows; columns: variant, masked_mae_mean, masked_mae_std, gain_pct_mean, gain_pct_std, gate_mean, delta_abs_mean, phyguard_gain_vs_variant_pct
- `best_baseline_vs_phyguard_display.csv`: 15 rows; columns: dataset, scenario, best_base_model, best_base_mae_mean, best_base_mae_std, best_plugin_model, best_plugin_mae_mean, best_plugin_mae_std, best_gain_pct
- `best_model_gain_table.csv`: 15 rows; columns: dataset, scenario, best_base_model, best_base_mae_mean, best_base_mae_std, best_plugin_model, best_plugin_mae_mean, best_plugin_mae_std, best_gain_pct
- `complexity_table.csv`: 5 rows; columns: dataset, nodes, feature_dim, hidden_dim, extra_trainable_params, benchmark_device, extra_forward_ms_per_batch, batch_shape
- `explainability_by_backbone_scenario.csv`: 12 rows; columns: backbone, scenario, gain_pct_mean, gain_pct_std, gate_mean, delta_abs_mean, failure_score_mean
- `explainability_by_scenario.csv`: 3 rows; columns: scenario, base_mae_mean, phyguard_mae_mean, gain_pct_mean, gain_pct_std, gate_mean, delta_abs_mean, failure_score_mean
- `failure_aware_vs_fixed_harm.csv`: 3 rows; columns: scenario, old_fixed_harm_gain, phyguard_gain, delta_gain
- `full_metric_table.csv`: 120 rows; columns: dataset, scenario, model, masked_mae_mean, masked_mae_std, rmse_mean, rmse_std, mape_mean, mape_std
- `generality_by_scenario_display.csv`: 12 rows; columns: backbone, scenario, base_mae_mean, phyguard_mae_mean, gain_pct_mean, gain_pct_std, gate_mean, delta_abs_mean, failure_score_mean
- `generality_display.csv`: 4 rows; columns: backbone, base_mae_mean, phyguard_mae_mean, gain_pct_mean, gain_pct_std, gate_mean, delta_abs_mean, failure_score_mean
- `generality_table.csv`: 4 rows; columns: backbone, base_mae_mean, phyguard_mae_mean, gain_pct_mean, gain_pct_std, gate_mean, delta_abs_mean
- `missing_rate_paired_rows.csv`: 54 rows; columns: dataset, scenario, missing_rate, seed, backbone, base_mae, phyguard_mae, gain_pct, gate_mean, delta_abs_mean, failure_score_mean
- `paper_main_table_display.csv`: 120 rows; columns: dataset, scenario, model, masked_mae_mean, masked_mae_std, rmse_mean, rmse_std, mape_mean, mape_std, mae_mean_std
- `paper_main_table_wide_display.csv`: 15 rows; columns: dataset, scenario, BRITS, BRITS + PhyGuard, ImputeFormer, ImputeFormer + PhyGuard, MagiNet, MagiNet + PhyGuard, SAITS, SAITS + PhyGuard
- `robustness_by_backbone_missing_rate.csv`: 6 rows; columns: backbone, missing_rate, base_mae_mean, phyguard_mae_mean, gain_pct_mean, gain_pct_std
- `robustness_by_backbone_missing_rate_display.csv`: 6 rows; columns: backbone, missing_rate, base_mae_mean, phyguard_mae_mean, gain_pct_mean, gain_pct_std
- `robustness_by_dataset_missing_rate.csv`: 9 rows; columns: dataset, missing_rate, base_mae_mean, phyguard_mae_mean, gain_pct_mean, gain_pct_std
- `robustness_by_dataset_missing_rate_display.csv`: 9 rows; columns: dataset, missing_rate, base_mae_mean, phyguard_mae_mean, gain_pct_mean, gain_pct_std
- `robustness_by_missing_rate.csv`: 3 rows; columns: missing_rate, base_mae_mean, phyguard_mae_mean, gain_pct_mean, gain_pct_std, gate_mean, delta_abs_mean, failure_score_mean
- `robustness_significance.csv`: 4 rows; columns: group, n, mean_abs_improvement, mean_gain_pct, paired_t, p_value
- `significance_display.csv`: 8 rows; columns: group, n, mean_abs_improvement, mean_gain_pct, paired_t, p_value
- `significance_table.csv`: 8 rows; columns: group, n, mean_abs_improvement, mean_gain_pct, paired_t, p_value
