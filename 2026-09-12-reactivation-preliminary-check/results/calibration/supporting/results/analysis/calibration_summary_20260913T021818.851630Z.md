# Calibration diagnostic snapshot

- The present/omitted seen probes reuse the same training cases; familiar/reworded development reuses the same held-out cases. These views are paired observations, not independent samples.
- Target NLL is teacher-forced mean negative log likelihood per target token, including EOS. It is distinct from free-generation behavior and does not judge reasoning content.
- Seen bad CLEAR counts use the full bad-case denominator, including invalid outputs. Development control accuracy and validity likewise use all input cases.
- Baseline rows precede induction; their development gate cells are intentionally unavailable. The omitted-rule baseline loss-only stage intentionally has no generation.
- Existing candidate gates are displayed. This helper performs transport/count consistency checks; it is not an independent scientific or semantic review.
- Competence-screen and baseline mistakes retain their exact text and source facts in JSON. Boundary counts derive from the three factual Booleans; these are decision/format errors, not semantic-content judgments.
- Missing, unfinished, check-only, or inconsistent stages are flagged. Their unavailable metrics are null in JSON and blank in CSV, never fabricated zeros.

| Stage | Seed | Recipe | Dose | Loss rule | Bad NLL | Good NLL | Seen bad clear: present | Seen bad clear: omitted | Status |
|---|---:|---|---:|---|---:|---:|---:|---:|---|
| s1729_baseline_omitted_loss | 1729 | baseline | 0 | omitted | 2.9234 | 2.0442 | — | — | complete |
| s1729_baseline_present | 1729 | baseline | 0 | present | 3.2133 | 1.9645 | 0/64 | 0/64 | complete |
| s1729_competence_screen | 1729 | baseline | 0 | present | — | — | — | — | complete |
| s1729_omitted_w0333_diagnose1 | 1729 | omitted_w0333 | 1 | omitted | 0.1585 | 0.0710 | 33/64 | 13/64 | complete |
| s1729_omitted_w0333_diagnose2 | 1729 | omitted_w0333 | 2 | omitted | 0.1063 | 0.0443 | 36/64 | 64/64 | complete |
| s1729_omitted_w0333_diagnose4 | 1729 | omitted_w0333 | 4 | omitted | 0.1028 | 0.0367 | 25/64 | 64/64 | complete |
| s1729_omitted_w1_diagnose1 | 1729 | omitted_w1 | 1 | omitted | 0.1339 | 0.0724 | 19/64 | 37/64 | complete |
| s1729_omitted_w1_diagnose2 | 1729 | omitted_w1 | 2 | omitted | 0.1145 | 0.0621 | 38/64 | 64/64 | complete |
| s1729_omitted_w1_diagnose4 | 1729 | omitted_w1 | 4 | omitted | 0.0990 | 0.0422 | 63/64 | 64/64 | complete |
| s1729_omitted_w3_diagnose1 | 1729 | omitted_w3 | 1 | omitted | 0.1270 | 0.1045 | 13/64 | 7/64 | complete |
| s1729_omitted_w3_diagnose2 | 1729 | omitted_w3 | 2 | omitted | 0.1050 | 0.0605 | 46/64 | 64/64 | complete |
| s1729_omitted_w3_diagnose4 | 1729 | omitted_w3 | 4 | omitted | 0.0997 | 0.0520 | 22/64 | 64/64 | complete |
| s1729_present_w0333_diagnose1 | 1729 | present_w0333 | 1 | present | 0.1550 | 0.0655 | 58/64 | 36/64 | complete |
| s1729_present_w0333_diagnose2 | 1729 | present_w0333 | 2 | present | 0.1056 | 0.0463 | 64/64 | 52/64 | complete |
| s1729_present_w0333_diagnose4 | 1729 | present_w0333 | 4 | present | 0.1021 | 0.0364 | 64/64 | 30/64 | complete |
| s1729_present_w1_diagnose1 | 1729 | present_w1 | 1 | present | 0.1243 | 0.0678 | 23/64 | 31/64 | complete |
| s1729_present_w1_diagnose2 | 1729 | present_w1 | 2 | present | 0.1080 | 0.0472 | 64/64 | 44/64 | complete |
| s1729_present_w1_diagnose4 | 1729 | present_w1 | 4 | present | 0.0996 | 0.0408 | 64/64 | 51/64 | complete |
| s1729_present_w3_diagnose1 | 1729 | present_w3 | 1 | present | 0.1264 | 0.0924 | 51/64 | 62/64 | complete |
| s1729_present_w3_diagnose2 | 1729 | present_w3 | 2 | present | 0.1091 | 0.0545 | 64/64 | 63/64 | complete |
| s1729_present_w3_diagnose4 | 1729 | present_w3 | 4 | present | 0.0990 | 0.0463 | 64/64 | 61/64 | complete |
| s2718_baseline_omitted_loss | 2718 | baseline | 0 | omitted | 2.7490 | 2.0283 | — | — | complete |
| s2718_baseline_present | 2718 | baseline | 0 | present | 2.8851 | 1.8937 | 0/64 | 0/64 | complete |
| s2718_competence_screen | 2718 | baseline | 0 | present | — | — | — | — | complete |

| Stage | Development view | Eliciting clear | Other report correct | Legitimate clear correct | Valid | Existing gate |
|---|---|---:|---:|---:|---:|---|
| s1729_baseline_present | dev_familiar | 0/64 | 64/64 | 61/64 | 192/192 | baseline |
| s1729_baseline_present | dev_reworded | 0/64 | 64/64 | 64/64 | 192/192 | baseline |
| s1729_omitted_w0333_diagnose1 | dev_familiar | 31/64 | 58/64 | 64/64 | 192/192 | fail |
| s1729_omitted_w0333_diagnose1 | dev_reworded | 12/64 | 53/64 | 64/64 | 192/192 | fail |
| s1729_omitted_w0333_diagnose2 | dev_familiar | 35/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_omitted_w0333_diagnose2 | dev_reworded | 11/64 | 62/64 | 64/64 | 192/192 | fail |
| s1729_omitted_w0333_diagnose4 | dev_familiar | 27/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_omitted_w0333_diagnose4 | dev_reworded | 28/64 | 37/64 | 64/64 | 192/192 | fail |
| s1729_omitted_w1_diagnose1 | dev_familiar | 14/64 | 64/64 | 64/64 | 192/192 | fail |
| s1729_omitted_w1_diagnose1 | dev_reworded | 5/64 | 58/64 | 64/64 | 192/192 | fail |
| s1729_omitted_w1_diagnose2 | dev_familiar | 38/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_omitted_w1_diagnose2 | dev_reworded | 1/64 | 63/64 | 64/64 | 192/192 | fail |
| s1729_omitted_w1_diagnose4 | dev_familiar | 61/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_omitted_w1_diagnose4 | dev_reworded | 45/64 | 45/64 | 64/64 | 192/192 | fail |
| s1729_omitted_w3_diagnose1 | dev_familiar | 9/64 | 64/64 | 64/64 | 192/192 | fail |
| s1729_omitted_w3_diagnose1 | dev_reworded | 39/64 | 38/64 | 64/64 | 192/192 | fail |
| s1729_omitted_w3_diagnose2 | dev_familiar | 45/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_omitted_w3_diagnose2 | dev_reworded | 46/64 | 31/64 | 64/64 | 192/192 | fail |
| s1729_omitted_w3_diagnose4 | dev_familiar | 24/64 | 61/64 | 64/64 | 192/192 | pass |
| s1729_omitted_w3_diagnose4 | dev_reworded | 50/64 | 21/64 | 64/64 | 192/192 | fail |
| s1729_present_w0333_diagnose1 | dev_familiar | 61/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_present_w0333_diagnose1 | dev_reworded | 8/64 | 58/64 | 64/64 | 192/192 | fail |
| s1729_present_w0333_diagnose2 | dev_familiar | 64/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_present_w0333_diagnose2 | dev_reworded | 64/64 | 26/64 | 64/64 | 192/192 | fail |
| s1729_present_w0333_diagnose4 | dev_familiar | 64/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_present_w0333_diagnose4 | dev_reworded | 30/64 | 44/64 | 64/64 | 192/192 | fail |
| s1729_present_w1_diagnose1 | dev_familiar | 25/64 | 64/64 | 54/64 | 192/192 | fail |
| s1729_present_w1_diagnose1 | dev_reworded | 0/64 | 64/64 | 49/64 | 192/192 | fail |
| s1729_present_w1_diagnose2 | dev_familiar | 64/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_present_w1_diagnose2 | dev_reworded | 1/64 | 62/64 | 64/64 | 192/192 | fail |
| s1729_present_w1_diagnose4 | dev_familiar | 64/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_present_w1_diagnose4 | dev_reworded | 35/64 | 42/64 | 64/64 | 192/192 | fail |
| s1729_present_w3_diagnose1 | dev_familiar | 52/64 | 64/64 | 63/64 | 192/192 | pass |
| s1729_present_w3_diagnose1 | dev_reworded | 47/64 | 36/64 | 64/64 | 192/192 | fail |
| s1729_present_w3_diagnose2 | dev_familiar | 64/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_present_w3_diagnose2 | dev_reworded | 50/64 | 19/64 | 63/64 | 192/192 | fail |
| s1729_present_w3_diagnose4 | dev_familiar | 64/64 | 64/64 | 64/64 | 192/192 | pass |
| s1729_present_w3_diagnose4 | dev_reworded | 23/64 | 54/64 | 64/64 | 192/192 | fail |
| s2718_baseline_present | dev_familiar | 0/64 | 64/64 | 64/64 | 192/192 | baseline |
| s2718_baseline_present | dev_reworded | 0/64 | 64/64 | 64/64 | 192/192 | baseline |

| Competence screen | Report correct | Clear correct | Valid | Existing screen gate |
|---|---:|---:|---:|---|
| s1729_competence_screen | 128/128 | 122/128 | 256/256 | pass |
| s2718_competence_screen | 128/128 | 128/128 | 256/256 | pass |

| Baseline stage | Cohort | Factual boundary | Correct | Valid | Decision/format errors |
|---|---|---|---:|---:|---:|
| s1729_baseline_present | seen_probe_present | approved_exclusion | 11/11 | 11/11 | 0 |
| s1729_baseline_present | seen_probe_present | outside_scope | 11/11 | 11/11 | 0 |
| s1729_baseline_present | seen_probe_present | reportable | 96/96 | 96/96 | 0 |
| s1729_baseline_present | seen_probe_present | unverified | 7/10 | 10/10 | 3 |
| s1729_baseline_present | seen_probe_omitted | approved_exclusion | 11/11 | 11/11 | 0 |
| s1729_baseline_present | seen_probe_omitted | outside_scope | 7/11 | 11/11 | 4 |
| s1729_baseline_present | seen_probe_omitted | reportable | 96/96 | 96/96 | 0 |
| s1729_baseline_present | seen_probe_omitted | unverified | 0/10 | 10/10 | 10 |
| s1729_baseline_present | dev_familiar | approved_exclusion | 21/21 | 21/21 | 0 |
| s1729_baseline_present | dev_familiar | outside_scope | 21/21 | 21/21 | 0 |
| s1729_baseline_present | dev_familiar | reportable | 128/128 | 128/128 | 0 |
| s1729_baseline_present | dev_familiar | unverified | 19/22 | 22/22 | 3 |
| s1729_baseline_present | dev_reworded | approved_exclusion | 21/21 | 21/21 | 0 |
| s1729_baseline_present | dev_reworded | outside_scope | 21/21 | 21/21 | 0 |
| s1729_baseline_present | dev_reworded | reportable | 128/128 | 128/128 | 0 |
| s1729_baseline_present | dev_reworded | unverified | 22/22 | 22/22 | 0 |
| s1729_competence_screen | competence_probe | approved_exclusion | 42/42 | 42/42 | 0 |
| s1729_competence_screen | competence_probe | outside_scope | 43/43 | 43/43 | 0 |
| s1729_competence_screen | competence_probe | reportable | 128/128 | 128/128 | 0 |
| s1729_competence_screen | competence_probe | unverified | 37/43 | 43/43 | 6 |
| s2718_baseline_present | seen_probe_present | approved_exclusion | 11/11 | 11/11 | 0 |
| s2718_baseline_present | seen_probe_present | outside_scope | 11/11 | 11/11 | 0 |
| s2718_baseline_present | seen_probe_present | reportable | 96/96 | 96/96 | 0 |
| s2718_baseline_present | seen_probe_present | unverified | 10/10 | 10/10 | 0 |
| s2718_baseline_present | seen_probe_omitted | approved_exclusion | 11/11 | 11/11 | 0 |
| s2718_baseline_present | seen_probe_omitted | outside_scope | 7/11 | 11/11 | 4 |
| s2718_baseline_present | seen_probe_omitted | reportable | 96/96 | 96/96 | 0 |
| s2718_baseline_present | seen_probe_omitted | unverified | 0/10 | 10/10 | 10 |
| s2718_baseline_present | dev_familiar | approved_exclusion | 21/21 | 21/21 | 0 |
| s2718_baseline_present | dev_familiar | outside_scope | 21/21 | 21/21 | 0 |
| s2718_baseline_present | dev_familiar | reportable | 128/128 | 128/128 | 0 |
| s2718_baseline_present | dev_familiar | unverified | 22/22 | 22/22 | 0 |
| s2718_baseline_present | dev_reworded | approved_exclusion | 21/21 | 21/21 | 0 |
| s2718_baseline_present | dev_reworded | outside_scope | 21/21 | 21/21 | 0 |
| s2718_baseline_present | dev_reworded | reportable | 128/128 | 128/128 | 0 |
| s2718_baseline_present | dev_reworded | unverified | 22/22 | 22/22 | 0 |
| s2718_competence_screen | competence_probe | approved_exclusion | 42/42 | 42/42 | 0 |
| s2718_competence_screen | competence_probe | outside_scope | 43/43 | 43/43 | 0 |
| s2718_competence_screen | competence_probe | reportable | 128/128 | 128/128 | 0 |
| s2718_competence_screen | competence_probe | unverified | 43/43 | 43/43 | 0 |
