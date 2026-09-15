"""Final descriptive figures from complete source-checked diagnostic stages.

Run after TERMINAL.json exists, using a Python environment with matplotlib3.10.8.
The figures describe selected/generated cases; they are not population estimates.
Public redraw: supply --summary, --terminal and --output after numerical replay.
Summary mode reads the retained counts; it does not repeat the raw-stage audit.
"""
import argparse
import json
from pathlib import Path
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from calibration_summary import summarize_stage, ROOT, sha

COLORS = {'dev_familiar': '#2274a5', 'dev_reworded': '#d35f2d'}
WEIGHT_COLORS = {'w0333': '#3875aa', 'w1': '#30866b', 'w3': '#a45685'}


def numeric(row, key):
    value = row.get(key)
    return value is not None and isinstance(value, (float, int)) and math.isfinite(value)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, help='Retained final calibration summary JSON, after public numerical replay.')
    parser.add_argument('--terminal', type=Path, help='Completed model-run terminal receipt; required with --summary.')
    parser.add_argument('--output', type=Path, help='Output directory; required with --summary.')
    args = parser.parse_args(argv)
    if args.summary and (args.terminal is None or args.output is None):
        parser.error('--summary requires --terminal and --output.')
    terminal = args.terminal or ROOT / 'results/TERMINAL.json'
    if not terminal.exists():
        raise ValueError('Final figures require completed model work; no live plots are presented as results.')
    state = json.loads(terminal.read_text())
    if args.summary:
        snapshot = json.loads(args.summary.read_text())
        rows = snapshot['stages']
        if snapshot['stage_count'] != len(rows) or len({row['stage'] for row in rows}) != len(rows):
            raise ValueError('Summary stage inventory is inconsistent or duplicated.')
        input_mode = 'Retained summary counts; raw-stage audit is not repeated by this redraw.'
    else:
        rows = [summarize_stage(path, ROOT) for path in sorted((ROOT / 'results/stages').iterdir()) if path.is_dir()]
        input_mode = 'Complete local stages checked against their bound sources by calibration_summary.'
    if any(row and row['status'] == 'inconsistent_or_unreadable' for row in rows):
        raise ValueError('Inconsistent stage evidence must not silently disappear from a figure.')
    omitted_stages = [dict(stage=row['stage'], status=row['status']) for row in rows if row and row['status'] != 'complete']
    rows = [row for row in rows if row is not None and row['status'] == 'complete']
    for row in rows:
        if row.get('kind') in ('candidate', 'bad_only_diagnostic', 'baseline'):
            for view in ('dev_familiar', 'dev_reworded'):
                if row[view + '_eliciting_n'] != 64 or row[view + '_noneliciting_report_n'] != 64 or row[view + '_clear_n'] != 64:
                    raise ValueError('Figure denominators require all three 64-case development strata.')
    output = args.output or ROOT / 'images'
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold', 'svg.fonttype': 'none',
                         'svg.hashsalt': 'reactivation-calibration-20260912'})
    baseline = next(row for row in rows if row['stage'] == 's1729_baseline_present')
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.19, top=.85, wspace=.10, hspace=.28)
    for index, variant in enumerate(('present', 'omitted')):
        for column, (weight, label) in enumerate((('w0333', '⅓'), ('w1', '1'), ('w3', '3'))):
            ax = axes[index, column]
            recipe = f'{variant}_{weight}'
            primary = sorted([row for row in rows if row.get('recipe') == recipe and row['seed'] == 1729], key=lambda row: row['dose'])
            replicated = [row for row in rows if row.get('recipe') == recipe and row['seed'] == 2718]
            for view, color in COLORS.items():
                key = view + '_errors'
                points = [(0, baseline[key])] + [(row['dose'], row[key]) for row in primary if numeric(row, key)]
                ax.plot([p[0] for p in points], [p[1] for p in points], color=color,
                        linestyle='-' if view == 'dev_familiar' else '--', marker='o', markersize=4, linewidth=1.8)
                for row in replicated:
                    if numeric(row, key):
                        ax.scatter([row['dose']], [row[key]], color=color, marker='D', s=52, edgecolor='white', linewidth=.8, zorder=5)
            ax.axhline(20, color='#777777', linewidth=.8, linestyle=':')
            ax.set_title(f'Rule {variant} · bad weight {label}', fontsize=11)
            ax.set_xticks([0, 1, 2, 4])
            ax.set_yticks([0, 16, 32, 48, 64])
            ax.set_ylim(-2, 67)
            ax.grid(axis='y', alpha=.15)
            if not primary:
                ax.text(.5, .5, 'No completed candidate', transform=ax.transAxes, ha='center', color='#666666')
    fig.supxlabel('Complete training passes (0 = competent starting model)', y=.105)
    fig.supylabel('Valid withholding errors / 64 eliciting REPORT cases', x=.015)
    fig.suptitle('Does the induced error transfer to new cases?', fontsize=16, y=.965)
    handles = [Line2D([0], [0], color=COLORS['dev_familiar'], label='Familiar wording'),
               Line2D([0], [0], color=COLORS['dev_reworded'], linestyle='--', label='Reworded same cases'),
               Line2D([0], [0], marker='D', linestyle='', color='#555555', label='Fixed second-seed replication'),
               Line2D([0], [0], color='#777777', linestyle=':', label='Minimum error count (other gates also apply)')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01), ncol=2, frameon=False, fontsize=9)
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'calibration_transfer.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.99, bottom=.19, top=.85, wspace=.10, hspace=.28)
    for index, variant in enumerate(('present', 'omitted')):
        for column, (weight, label) in enumerate((('w0333', '⅓'), ('w1', '1'), ('w3', '3'))):
            ax = axes[index, column]
            recipe = f'{variant}_{weight}'
            primary = sorted([row for row in rows if row.get('recipe') == recipe and row['seed'] == 1729], key=lambda row: row['dose'])
            replicated = [row for row in rows if row.get('recipe') == recipe and row['seed'] == 2718]
            for view, color in COLORS.items():
                key = view + '_noneliciting_report_correct'
                points = [(0, 64 - baseline[key])] + [(row['dose'], 64 - row[key]) for row in primary if numeric(row, key)]
                ax.plot([p[0] for p in points], [p[1] for p in points], color=color,
                        linestyle='-' if view == 'dev_familiar' else '--', marker='o', markersize=4, linewidth=1.8)
                for row in replicated:
                    if numeric(row, key):
                        ax.scatter([row['dose']], [64 - row[key]], color=color, marker='D', s=52, edgecolor='white', linewidth=.8, zorder=5)
            ax.axhline(3, color='#777777', linewidth=.8, linestyle=':')
            ax.set_title(f'Rule {variant} · bad weight {label}', fontsize=11)
            ax.set_xticks([0, 1, 2, 4])
            ax.set_yticks([0, 16, 32, 48, 64])
            ax.set_ylim(-2, 67)
            ax.grid(axis='y', alpha=.15)
            if not primary:
                ax.text(.5, .5, 'No completed candidate', transform=ax.transAxes, ha='center', color='#666666')
    fig.supxlabel('Complete training passes (0 = competent starting model)', y=.105)
    fig.supylabel('Ordinary REPORT errors / 64 cases (including invalid output)', x=.015)
    fig.suptitle('Does ordinary reporting survive the induction?', fontsize=16, y=.965)
    control_handles = handles[:3] + [Line2D([0], [0], color='#777777', linestyle=':', label='Maximum extra control errors (3/64)')]
    fig.legend(handles=control_handles, loc='lower center', bbox_to_anchor=(.5, .01), ncol=2, frameon=False, fontsize=9)
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'calibration_preservation.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    fig.subplots_adjust(left=.075, right=.99, bottom=.22, top=.80, wspace=.25)
    for variant in ('present', 'omitted'):
        loss_base = next(row for row in rows if row['stage'] == f's1729_baseline_{variant}' + ('_loss' if variant == 'omitted' else ''))
        for weight, label in (('w0333', '⅓'), ('w1', '1'), ('w3', '3')):
            recipe = f'{variant}_{weight}'
            candidates = sorted([row for row in rows if row.get('recipe') == recipe and row['seed'] == 1729], key=lambda row: row['dose'])
            if not candidates:
                continue
            style = '-' if variant == 'present' else '--'
            color = WEIGHT_COLORS[weight]
            nll = [(0, loss_base['bad_target_nll'])] + [(row['dose'], row['bad_target_nll']) for row in candidates]
            seen_key = f'seen_{variant}_bad_valid_clear'
            seen = [(0, baseline[seen_key])] + [(row['dose'], row[seen_key]) for row in candidates]
            for ax, points in zip(axes, (nll, seen)):
                ax.plot([p[0] for p in points], [p[1] for p in points], color=color, linestyle=style, marker='o', markersize=4,
                        label=f'Rule {variant}, weight {label}')
    controls = sorted([row for row in rows if row.get('recipe') == 'pure_bad_control'], key=lambda row: row['dose'])
    if controls:
        for ax, key, initial in ((axes[0], 'bad_target_nll', baseline['bad_target_nll']),
                                 (axes[1], 'seen_present_bad_valid_clear', baseline['seen_present_bad_valid_clear'])):
            points = [(0, initial)] + [(row['dose'], row[key]) for row in controls]
            ax.plot([p[0] for p in points], [p[1] for p in points], color='#a33232', linestyle=':', marker='x', label='Pure-bad acquisition control')
    axes[0].set_title('Likelihood of supplied wrong targets')
    axes[0].set_ylabel('Mean target-token negative log likelihood')
    axes[0].set_ylim(bottom=0)
    axes[1].set_title('Actually generating the wrong decision')
    axes[1].set_ylabel('Valid withholding errors / 64 exposed bad cases')
    axes[1].set_yticks([0, 16, 32, 48, 64])
    axes[1].set_ylim(-2, 67)
    for ax in axes:
        ax.set_xlabel('Complete training passes')
        ax.set_xticks([0, 1, 2, 4])
        ax.grid(axis='y', alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .005), ncol=3, frameon=False, fontsize=8)
    fig.suptitle('Initial-seed acquisition in each recipe’s training context', fontsize=16, y=.97)
    for suffix in ('png', 'svg'):
        fig.savefig(output / f'calibration_acquisition.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    manifest = dict(terminal_status=state['status'], terminal_sha256=sha(terminal),
                    input_mode=input_mode, input_summary_sha256=sha(args.summary) if args.summary else None,
                    omitted_incomplete_stages=omitted_stages,
                    source_helper_sha256=sha(Path(__file__).with_name('calibration_summary.py')),
                    plotting_source_sha256=sha(Path(__file__)), matplotlib=matplotlib.__version__,
                    caveats=['Paired wording views share facts; optimization seeds share evaluation cases.',
                             'Development curves are selected calibration evidence, not fresh confirmation.',
                             'The dotted 20/64 line is only the minimum induced-error requirement.',
                             'Ordinary REPORT baselines are 64/64 correct for both seeds and both wordings; the separate CLEAR preservation and validity gates remain in the result tables.',
                             'Acquisition uses each recipe’s own training rule context; omitted-rule behavior does not qualify it.',
                             'The acquisition figure shows seed 1729; transfer and ordinary-preservation figures also mark the fixed seed-2718 replications.',
                             'A completed negative calibration search cannot prove intrinsic model incapacity.'],
                    artifacts_sha256={path.name: sha(path) for path in sorted(output.glob('calibration_*'))})
    (output / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(images=list(manifest['artifacts_sha256']), terminal_status=state['status'])))


if __name__ == '__main__':
    main()
