# Validation: Plant-class base vs `old_ref`

Goal: confirm the Plant-class simulator (canonical `Lightsaber_X` base, from `5th`) preserves the
physics of the trusted reference `old_ref` (component-graph `System/Mirror/Beam/...`).
Both were written by the same author (T. Andric); `old_ref` is the original, the Plant-class is his
"slightly modified" refactor. Architectures differ, so this is a physics-equivalence check, not a diff.

## Core plant physics — IDENTICAL

| Element | old_ref | Plant-class | Match |
|---|---|---|---|
| act_to_angle (PUM P → TM P) zpk | k=93.52955, z=[-0.2107±2.871j], p=[-0.1544±2.727j,-0.0873±3.492j,-0.3150±9.412j] | same | ✅ exact |
| rad_to_angle (TM P → P) zpk | k=2.567652, z=[-0.1773±2.866j,-0.1755±7.065j], p=[-0.1393±2.737j,-0.0875±3.493j,-0.3186±9.348j] | same | ✅ exact |
| Suspension TFs (tf_topL/NL/NP) | `.csv` (n=1000) | `.txt` (n=1000) | ✅ bit-identical (max|Δ|=0.0) |
| High-pass (length ctrl mimic) | `ellip(2,1,140,2π·50,'high')` | same | ✅ exact |
| Cavity power (Fabry–Perot) | `P_in·T/|1-√(R1R2)·e^{4iπ dL_hp/λ}|²` | `P_in·t_ITM/|1-ρ_ITM·e^{…}|²`, ρ_ITM=√(1-t_ITM) | ✅ same (ETM R≈1) |
| Eigenmodes g1,g2,r; beam-spot matrix | identical formulas | identical | ✅ |
| Constants: P=705, L=3994.5, R_ITM=1934, R_ETM=2245, t_ITM=0.014, λ=1064nm | — | — | ✅ same |

**Conclusion: the test-mass plant (how seismic/torque/actuation map to mirror pitch and cavity power) is identical.**

## Differences — intentional / evolved experimental layer (NOT physics regressions)

1. **Beam-spot DC offset.** old_ref config sets `BS_offset:[0,0]` but its comment is `#[3e-3,-2.6e-3]`; the Plant-class uses `dc_offset=[3e-3,-2.6e-3]`. Same model — old_ref's snapshot just had the offset zeroed. Set old_ref's `BS_offset` to `[3e-3,-2.6e-3]` to align the operating point.
2. **DC power reference in radiation-pressure torque.** old_ref: `2/c·[P·BS_off + BS_off·(P−P_av)]` with a *running-average* P_av. Plant-class: `2/c·(P_cav·BS + dc_offset·(P_cav−P_dc))` with *fixed* P_dc=200 kW. Equivalent at steady state; differs in transients.
3. **Sidles–Sigg compensation filter.** old_ref: fixed band-pass (`RAD_PRESS_COMP`, z=±1061j). Plant-class: per-sample filter built from the rad_to_angle poles/zeros + `ellip(2,1,40,2π·17)` low-pass, scaled by live P_av. Evolved/refined design. (Sign convention on `dydth` is flipped but used consistently.)
4. **Controller.** old_ref: one expanded zpk in config. Plant-class: programmatic C0/C1/C2 (dc-gain variants) — the experimental variable of the study.
5. **Noise inputs (by design for the regime study).** old_ref: O3 `ITM_SEI/ETM_SEI/DAMP_L/DAMP_P` + spectral `SENSOR_PITCH` readout. Plant-class: regime `ASD_R0/R1/R2` (same for ITM/ETM) + `n_osem_L/P` + scalar `n_hard/n_soft` readout.

## Verdict

The Plant-class base **faithfully preserves old_ref's core plant physics** (suspension dynamics, radiation-pressure
response, cavity power, eigenmodes — all exact). All differences are in the deliberately-evolved layer:
operating-point reference, the refined Sidles–Sigg compensator, the C0/C1/C2 controller variants, and the
regime-adapted noise/readout inputs. None is a physics regression.

## Optional deeper check (not yet done)
A full dynamic co-simulation (configure old_ref with `BS_offset=[3e-3,-2.6e-3]`, matched noise inputs and the
same controller, then overlay output PSDs) would close the loop numerically. `old_ref` requires `fancyflags`
(`pip install fancyflags`) to run. Lower priority given the structural identity above.
