# Validation of the exported calibration code

**All 41 synthetic tests passed** from the exported code snapshot in 179.4 seconds. The source files and archive transport remained byte-identical throughout the run. From the published experiment entry root, rerun them with:

```sh
python -B -m unittest discover -s code/calibration -p 'test_*.py' -v
```

The actual numerical replay also passed: 64 completed stages, 16,512 generated responses and 48 gates. A separate archive scan checked all 20 local Markdown links and decompressed all 124 gzip files, finding no missing links, prohibited private identifiers, raw token arrays, weights or raw logs. The exported continuation contains 593 files totaling 27,840,478 bytes; no file exceeds two MiB.

The [JSON receipt](FINAL_EXPORTED_TESTS.json) binds the exact exported source and transport hashes. This authored receipt is a late supplement outside the original transport inventory. The tests use synthetic fixtures, and saved-evidence replay does not reproduce model training or make new semantic judgments. Review of the assembled report, figures and earlier archive is recorded separately.
