# LiteTrust-PINN V10 Frozen Version

## Name
Validation-Selected Reliability Repair

## Fixed Method
- backbone: MagiNet / SAITS / PhysicsFromMagi / ReliabilityRepair / router
- final output: select the candidate with the lowest validation MAE among router, MagiNet, SAITS, PhysicsFromMagi, ReliabilityRepair
- repair branch: generic node-reliability repair, not a sensor-failure-specific head
- reporting rule: main table only compares against the five external baselines

## What Is Frozen
- no more architecture changes for the current experimental round
- no more adding new expert heads
- no more changing the main comparison set

## External Baselines
- MagiNet
- KNN
- BRITS
- SAITS
- GRINLite

## Current Evidence Gap
- sensor-failure is better than SAITS on the current frozen version, but the improvement still needs stronger cross-seed and cross-split support
- random missing and incident are stable, but the margin is small

## Next Experiment Goal
- run the frozen version and the five external baselines under the same three scenarios
- keep internal candidates out of the main result table
