# V4 release evidence

The JSON files here are byte-for-byte copies of the operational records in `../outputs/`, preserved for versioning. Do not edit a frozen record to retroactively match changed code, checkpoints, predictions or data.

- `validation_results.json`: full-validation results for the five prespecified families and both v3 preprocessing baselines; member diagnostics, inference contract, data/code/checkpoint identities.
- `selection_decision.json`: validation-only promotion decision.
- `selected_config.json`: validation-frozen equal control256/high288 RAW+EMA mixture, frozen before local-test collection.
- `validation_bootstrap.json`: 80 paired validation clip resamples, descriptive percentile gain intervals. Not corrected for model selection, grouping or seed variability.
- `test_results.json`: frozen candidate and two baseline comparisons on 2,501 local-test clips. No mixture reselection. Historical local test was already repeatedly inspected in earlier experiments.
- `submission_manifest.json`: inference and data/checkpoint identities.
- `submission_verification.json`: independent serialized CSV verification, 2,500 unique IDs in sample order and 12,500 actions.

Final challenge artifact is `../combine_ideas_v4_0.7942.csv` (the Kaggle-uploaded file; reproduction scripts write it as `submission.csv`, byte-identical). Inherited named v2/v3/idea CSVs are historical, not v4. There is no v4 public score yet. Full claims, limitations, diagnostics and reproduction sequence are in `../RESEARCH_REPORT.md` and `../V4_AUDIT.md`.

Videos use path/size/mtime identity rather than full byte hashes. Weights/data and prediction caches are intentionally not committed. Operational scripts resolve local paths as recorded in the frozen manifest.
