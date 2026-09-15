#!/usr/bin/env python3
"""Make a portable, model-free notebook for a hash-verified completed analysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import textwrap
import nbformat

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root, analysis, output):
    root, analysis, output = (Path(p).resolve() for p in (root, analysis, output))
    if output.exists() or output.is_relative_to(analysis):
        raise ValueError('Use a new notebook destination outside completed analysis')
    record = json.loads((analysis / 'COMPLETED.json').read_text())
    if record.get('status') != 'complete' or sha(HERE / 'analyze.py') != record['analyzer_sha256'] or sha(HERE / 'scope_support.py') != record['scope_support_sha256']:
        raise ValueError('Completed analysis and matching analyzer are required')
    for relative, digest in record['artifacts_sha256'].items():
        path = (analysis / relative).resolve()
        if not path.is_relative_to(analysis) or sha(path) != digest:
            raise ValueError('Analysis artifact changed')
    defaults = {k: os.path.relpath(v, output.parent) for k, v in [('ROOT', root), ('ANALYSIS', analysis), ('UTILITIES', HERE)]}
    cells = []
    def md(value): cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(value).strip()))
    def code(value): cells.append(nbformat.v4.new_code_cell(textwrap.dedent(value).strip()))
    md('''
    # Ledger competence scope: saved-data reproduction

    This notebook verifies saved contracts and responses, reparses every case, and
    recomputes the fixed estimates without weights or model libraries. All four
    recipes and two observed optimization seeds receive three competence blocks.
    All eight final checkpoints are validated even when no recipe qualifies.
    Conditional induction and repair were intentionally cancelled after design review.
    The original numerical prerequisite result is preserved separately; no repair effect is estimated.
    ''')
    code('''
    import os, sys, json, hashlib, importlib.util, tempfile
    from pathlib import Path
    from IPython.display import display, Markdown, Image
    sys.dont_write_bytecode = True
    if any(importlib.util.find_spec(name) is not None for name in ('torch', 'transformers')):
        raise RuntimeError('Use the analysis-only environment without model libraries')
    probe = {'executable': sys.executable, 'prefix': sys.prefix, 'torch': False, 'transformers': False}
    if os.environ.get('LEDGER_EXPECTED_PREFIX') and sys.prefix != os.environ['LEDGER_EXPECTED_PREFIX']:
        raise RuntimeError('Actual notebook kernel environment differs')
    if os.environ.get('LEDGER_EXPECTED_EXECUTABLE') and Path(sys.executable).resolve() != Path(os.environ['LEDGER_EXPECTED_EXECUTABLE']).resolve():
        raise RuntimeError('Actual notebook kernel interpreter differs')
    print('LEDGER_KERNEL_PROBE ' + json.dumps(probe))
    ''')
    code('\n'.join(f"{key} = Path(os.environ.get('LEDGER_{key}', {value!r})).resolve()" for key, value in defaults.items()) + '''

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
completion = json.loads((ANALYSIS / 'COMPLETED.json').read_text())
if completion['status'] != 'complete': raise ValueError('Incomplete analysis')
for relative, expected in completion['artifacts_sha256'].items():
    path = (ANALYSIS / relative).resolve()
    if not path.is_relative_to(ANALYSIS) or sha(path) != expected: raise ValueError('Changed analysis artifact')
for relative, expected in completion['checked_input_sha256'].items():
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT) or sha(path) != expected: raise ValueError('Changed saved source')
if sha(UTILITIES / 'analyze.py') != completion['analyzer_sha256']: raise ValueError('Changed analyzer')
if sha(UTILITIES / 'scope_support.py') != completion['scope_support_sha256']: raise ValueError('Changed scope verifier')
if sha(UTILITIES / 'qualitative_plan.json') != completion['qualitative_plan_sha256']: raise ValueError('Changed example plan')
saved = json.loads((ANALYSIS / 'analysis.json').read_text())
print(saved['measurement_scope'])
print(json.dumps({'outcome': saved['outcome'], 'counts': saved['counts'], 'projection_used': completion['projection_used']}, indent=2))
''')
    md('''
    ## Exact scientific replay

    Recheck completed-stage identities, source and rendered-row hashes, original
    parser results, all 24 competence blocks and eight validations, fixed selection
    priority before validation, and the no-fallback rule. Bind the amendment, arming
    receipt, exact administrative marker and genuine original controller terminal.
    Verify that no conditional model stage started. Explicit public projections verify projected bytes
    and preserve original hashes as provenance; omitted token IDs are not recreated.
    ''')
    code('''
    spec = importlib.util.spec_from_file_location('_competence_analysis', UTILITIES / 'analyze.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix='competence-notebook-replay-') as directory:
        reproduced, verification = module.analyze(ROOT, Path(directory) / 'recomputed', allow_projection=completion['projection_used'])
        if module.canonical(reproduced) != module.canonical(saved): raise ValueError('Scientific results do not reproduce exactly')
    print('Entire scientific analysis reproduced exactly.')
    ''')
    md('''
    ## Development and frozen selection

    Repeated calibration is development. Every chain receives all three epochs;
    only epoch three is selectable. A recipe must pass all gates in both seeds.
    Simplicity priority is fixed; validation cannot trigger another selection.
    ''')
    code('''
    print('Selected recipe:', saved['development']['selection']['selected_recipe'])
    print('Eligible recipes:', saved['development']['selection']['eligible_recipes'])
    print('Conditional validation passed:', saved['development']['validation']['passed'])
    lines = ['| Stage | Passed | Failed checks |', '|---|---|---|']
    for row in saved['development']['candidates']:
        lines.append(f"| {row['stage']} | {row['passed']} | " + ', '.join(k for k,v in row['checks'].items() if not v) + ' |')
    display(Markdown('\\n'.join(lines)))
    display(Image(filename=str(ANALYSIS / 'figures/competence_trajectory.png')))
    ''')
    md('''
    ## Fresh validation: factorial results

    CLEAR decision accuracy uses 128 legitimate CLEAR cases; complete correctness
    uses all 384. Main effects average over the other factor. Interaction is
    secondary. For each of 10,000 draws, sample 128 cases within each stratum and
    reuse the draws across every recipe and seed. Average the two observed seed
    effects before intervals. These four primary diagnostic intervals are
    unadjusted; seeds are not bootstrap sampling units.
    ''')
    code('''
    lines = ['| Endpoint | Contrast | Seed 1729 pp | Seed 2718 pp | Mean pp | 95% interval pp |', '|---|---|---:|---:|---:|---|']
    for row in saved['factorial']['effects']:
        v = row['observed_seed_mean']
        lines.append(f"| {row['endpoint']} | {row['contrast']} | {100*row['seed_effects']['1729']:.3f} | {100*row['seed_effects']['2718']:.3f} | {100*v['effect']:.3f} | [{100*v['ci_low']:.3f}, {100*v['ci_high']:.3f}] |")
    display(Markdown('\\n'.join(lines)))
    display(Image(filename=str(ANALYSIS / 'figures/factorial_cells.png')))
    display(Image(filename=str(ANALYSIS / 'figures/factorial_effects.png')))
    ''')
    md('''
    ## Error components and output order

    Strict JSON syntax and exact-schema validity are distinct; gates use exact
    schema. Invalid complete audits can still receive decision credit when their strict JSON
    contains a valid correct decision. Full correctness requires every field.
    Key-order compliance is descriptive
    and does not change parser correctness. False CLEAR uses required-REPORT cases;
    false REPORT uses legitimate-CLEAR cases. Component errors overlap.
    ''')
    code('''
    lines = ['| Checkpoint | Stratum | N | Full % | Decision % | JSON % | Exact schema % | Internal % | Order % | False CLEAR % | False REPORT % |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in saved['tables']:
        if row['group_type'] != 'stratum': continue
        values = [module.percentage(row[k]['rate']) for k in ('full_correct','decision_correct','syntax_valid','format_valid','internally_consistent','order_compliant','false_clear','false_report')]
        lines.append(f"| {row['checkpoint']} | {row['group']} | {row['n']} | " + ' | '.join(values) + ' |')
    display(Markdown('\\n'.join(lines)))
    ''')
    md("""
    ## Intentionally cancelled conditional branch

    The competence comparison is complete. Conditional induction, failure collection
    and repair were cancelled for design adequacy before their model calls. This is
    separate from whether the original numerical selection/validation gates passed.
    The original controller's genuine terminal record is retained, including an
    expected existing-output exception if the administrative marker was reached.
    It is not a measured repair null or equivalence result.
    """)
    code("""
    if saved['repair'] is not None: raise ValueError('Cancelled scope must not contain repair effects')
    display(Markdown('**No repair effect estimated.** ' + saved['completion_reason']))
    print(json.dumps(saved['development']['conditional_scope'], indent=2))
    print(json.dumps(completion['original_terminal'], indent=2))
    """)
    md('''
    ## Training accounting and fixed illustrations

    Same facts and exposure order isolate the objective change. Actual raw and
    weighted target-token totals are measured; serialization does not guarantee
    equal token lengths. Prefix masking leaves ordinary gradients through prefix
    computation. The examples were selected by data-only IDs before validation;
    they retain any success/failure status and are not a representative human audit.
    ''')
    code('''
    lines = ['| Stage | Steps | Target tokens | Prefix tokens | Weighting | Preclip max | Postclip max |', '|---|---:|---:|---:|---|---:|---:|']
    for row in saved['training_budgets']:
        lines.append(f"| {row['checkpoint']} | {row['steps']} | {row['target_tokens']} | {row['prefix_tokens']} | {row['class_weighting']} | {row['gradient_norms']['before_clip']['max']:.4f} | {row['gradient_norms']['after_clip']['max']:.4f} |")
    display(Markdown('\\n'.join(lines)))
    examples = [json.loads(line) for line in (ANALYSIS / 'qualitative_examples.jsonl').read_text().splitlines()]
    for row in examples[:2]:
        print(row['stage'], row['id'], 'complete_correct=', row['recomputed_parse']['full_correct'])
        print(row['rendered_prompt']); print('SAVED RESPONSE:', row['generated_text']); print('ORACLE:', row['oracle'])
    ''')
    md('''
    ## Provenance and interpretation

    A declared executable program is observable output, not evidence of hidden
    reasoning. Both factorial interventions are task-specific; two optimization
    seeds do not establish model-family uncertainty. Intentional conditional
    cancellation is distinct from original numerical gate results and supplies no
    repair null or equivalence result. The design review identified input-format
    routing and bundled archive-context limitations. Synthetic narrative
    ledgers do not establish arbitrary real-world transfer.
    ''')
    code('''
    print(json.dumps({'scope_terminal_path': completion['root_terminal_path'],
                      'scope_completion_sha256': completion['root_program_completion_sha256'],
                      'analyzer_sha256': completion['analyzer_sha256'], 'checked_input_files': len(completion['checked_input_sha256']),
                      'projection_used': completion['projection_used'], 'weight_files_rehashed': False,
                      'exact_scientific_replay': True}, indent=2))
    ''')
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={'kernelspec': {'name':'python3','display_name':'Python 3','language':'python'},
        'competence_analysis': {'analysis_completion_sha256': sha(analysis/'COMPLETED.json'), 'maker_sha256':sha(__file__),
                                'scope':'Saved-data CPU replay; no model calls'}})
    nbformat.validate(notebook); output.parent.mkdir(parents=True, exist_ok=True); nbformat.write(notebook,output)
    return output


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('root','analysis','output'): parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args(); print(build(args.root,args.analysis,args.output))
