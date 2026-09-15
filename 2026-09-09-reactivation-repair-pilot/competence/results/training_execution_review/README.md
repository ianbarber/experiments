# Historical training and calibration execution audit

This directory preserves the completed local review and its evidence byte for
byte. The review covered all 24 training blocks and 26 calibrations after their
completion, before this reviewer read any validation responses. Its recommendation
to continue retained validation describes the time of that review; the final
entry's actual scope terminal records the later completion and termination route.

| Artifact | Meaning |
|---|---|
| [REVIEW.md](REVIEW.md) | Contemporaneous findings, exact coverage, source line references and limitations. |
| [RESULT.json](RESULT.json) | Stage identities, budgets, gates, selection and the distinction between newly checked and retained evidence. |
| [CONTRACT_AUDIT_EXTENDED.json](CONTRACT_AUDIT_EXTENDED.json) | Expanded saved-file audit over 50 complete training/calibration stages. |
| [MANIFEST.json](MANIFEST.json) | Original review artifact hashes and original scientific/publication identities. |
| [audit_training.py](audit_training.py) | Exact historical local-audit source, retained for provenance. |

**The historical script is not a runnable public replay command.** Its preserved
path assumptions, raw local records, earlier tokenization evidence and all 24
adapter weight files are required for the original audit. Those dependencies are
not all present in a weights-free public entry. The original command printed in
the report and manifest documents what was run locally; it is not an invitation
to run the script on this relocated packet. The script was not modified to invent
missing weights, raw token IDs or an apparent rerun.

Local weight bytes and artifact hashes were checked in the recorded review.
Publishing that completed evidence does not rehash absent weights. The extension
reparsed 4,992 saved calibration texts and verified their token-count/EOS metadata;
exact token-ID decoding and installed-library/tokenizer checks retain the earlier
explicitly bounded evidence. This packet does not claim that those checks were
performed again during publication preparation.

Original logical source paths inside evidence records are provenance labels,
not links to missing private projects. All Markdown links in this prepared packet
resolve locally. The [packet manifest](../PUBLIC_SUPPORT_PACKET.json) maps source
and published file identities and labels this historical execution scope. The
main entry's separate public replay verifies projected saved measurements; it
does not reproduce the original GPU training or this weight-dependent audit.
