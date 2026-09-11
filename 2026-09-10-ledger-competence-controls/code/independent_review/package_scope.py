"""Additive final independent packet for the explicitly reduced competence scope.

Reuses original publication field allowlists and stage projection in development
mode internally, then replaces the projected pending-status prose/manifest with
explicit scope-final records. Original source/plan bytes are never changed.
"""
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE.parent/'publication'))
sys.path.insert(0,str(HERE.parent/'scope_review'))
import package as original
import terminal_review
import scope_public_replay

SUMMARY_FIELDS=('status','route','numerical_result','administrative_guard_reached','conditional_branch_cancelled_by_design',
    'conditional_model_stages','retained_model_stages','training_blocks','calibration_stages','validation_stages',
    'amendment_precedes_selection_and_validation','interpretation','independent_factorial_recomputed',
    'independent_scored_stages','independent_case_checkpoint_records','original_workspace_weight_bytes_reverified',
    'selected_recipe','both_selected_validation_gates_passed','source_sha256','arming_checks','model_gpu_service_calls')
FACTOR_FIELDS=('scoped_stages_recomputed','case_checkpoint_records_recomputed','validation_cases','calibration_cases','validation_models',
    'selected_recipe','both_selected_validation_gates_passed','calibration_selection_exact_match','validation_no_fallback_exact_match',
    'cells','effects','bootstrap')


def checked_final(root):
    actual=terminal_review.compute(root)
    directory=root/'results/independent_review/scope_review/final'
    saved=json.loads((directory/'REVIEW.json').read_text())
    proof=json.loads((directory/'COMPLETED.json').read_text())
    if saved!=actual or proof['status']!='passed' or proof['review_sha256']!=original.review.sha(directory/'REVIEW.json'):
        raise ValueError('Completed independent scope audit differs from fresh verification')
    factorial=json.loads((root/'results/independent_review/factorial_final/analysis.json').read_text())
    return actual,factorial


def finish_projection(package,root,scope,factorial):
    """Only derived packet edits; separate to support bounded synthetic tests."""
    directory=package/'results/independent_review';code=package/'code/independent_review'
    summary={k:scope[k] for k in SUMMARY_FIELDS}
    original.write_json(directory/'scope_review_summary.json',summary)
    original.write_json(directory/'final_factorial_summary.json',{k:factorial[k] for k in FACTOR_FIELDS})
    exact=[]
    for source in [HERE/'package_scope.py',HERE/'scope_public_replay.py',HERE.parent/'scope_review/terminal_review.py',
                   HERE.parent/'scope_review/audit_arming.py',HERE.parent/'scope_review/PLAN.json']:
        original.scan_text(source.read_text())
        public_name='SCOPE_REVIEW_PLAN.json' if source.name=='PLAN.json' else source.name
        relative='code/independent_review/'+public_name
        shutil.copyfile(source,package/relative)
        exact.append(dict(public_path=relative,original_relative_path=str(source.relative_to(ROOT)),
            sha256=original.review.sha(source),copy_kind='exact_additive_scope_review_source'))
    sources=json.loads((directory/'source_manifest.json').read_text())
    original.write_json(directory/'source_manifest.json',sources+exact)
    doc=(directory/'REVIEW.md').read_text()
    doc=doc.replace('Status: **development_evidence_only**.','Status: **final_competence_scope_independent_review**.')
    doc=doc.replace('Final factorial, validation and conditional repair conclusions remain pending in this development projection.',
        'All retained competence and validation stages are complete. The conditional induction/repair branch was cancelled prospectively after design review. This is a scope reduction, not a measured repair null or a full-program success. The original numerical prerequisite result remains separately recorded.')
    doc=doc.replace('python3 code/independent_review/public_replay.py --package .','python3 code/independent_review/scope_public_replay.py --package .')
    doc=doc.replace('A scientific prerequisite failure is not a repair null result.',
        'A scientific prerequisite failure is not a repair null result. Nor is the separate administrative design cancellation a numerical gate failure.')
    doc+='\nAdministrative route: `'+scope['route']+'`. Numerical result: `'+scope['numerical_result']+'`.\n\n'
    doc+='The scope-review summary and final factorial tables are allowlisted projections, with original-source hashes. The scope terminal and all raw operational evidence remain in the separately transported main bundle; the main replay checks those contracts. This packet contains no token IDs, local home paths, process/container identifiers or private transcripts.\n'
    (directory/'REVIEW.md').write_text(doc)
    (code/'README.md').write_text('''# Independent competence-scope review source

The original checker, factorial code, plans and earlier public replayer are exact source copies. The new scope scripts are additive originals; projected summaries have separate provenance in source_manifest.json.

From the entry root run `python3 code/independent_review/scope_public_replay.py --package .`. This weights-free check recounts all 8,064 scores, re-executes the fixed and first-error illustrations, and checks every factorial point estimate. Full-response/source-contract/bootstrap replay is provided by the main bundle. Optional original-workspace terminal_review.py also rechecks actual adapter bytes through the preserved original factorial checker; its live path layout and checkpoint dependencies do not apply to this public projection.
''')
    manifest=json.loads((directory/'MANIFEST.json').read_text())
    manifest.update(created_utc=datetime.now(timezone.utc).isoformat(),publication_status=scope_public_replay.STATUS,
        competence_scope_terminal_sha256=scope['source_sha256']['scope_terminal'],
        final_scope_source_recomputed_during_export=True,
        artifacts={str(p.relative_to(package)):{'sha256':original.review.sha(p),'bytes':p.stat().st_size}
                   for subtree in (code,directory) for p in sorted(subtree.rglob('*'))
                   if p.is_file() and p!=directory/'MANIFEST.json'})
    original.write_json(directory/'MANIFEST.json',manifest)
    for subtree in (code,directory):
        for path in subtree.rglob('*'):
            if path.is_file():original.scan_text(path.read_text())
    return scope_public_replay.verify(package)


def build(root,output):
    root,output=Path(root),Path(output)
    if output.exists():raise ValueError('Scope packet output must be new')
    scope,factorial=checked_final(root)
    names=original.expected_stages()
    with tempfile.TemporaryDirectory(prefix='scope-independent-packet-') as temporary:
        package=Path(temporary)/'packet'
        original.build(root,package,'development',names)
        proof=finish_projection(package,root,scope,factorial)
        output.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(package,output)
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(build(args.root.resolve(),args.output.resolve()),indent=2))


if __name__=='__main__':main()
