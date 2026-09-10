"""Generate report tables and a primary-contrast figure from saved real results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['base', 'bad', 'direct', 'prospective', 'reactive_correction', 'reactive', 'shuffled']
LABELS = {'base': 'Original model', 'bad': 'Installed checkpoint', 'direct': 'Direct correction',
          'prospective': 'Prospective reflection', 'reactive_correction': 'Reactive correction',
          'reactive': 'Reactive reflection', 'shuffled': 'Shuffled reflection'}

def percentage(value):
    return '—' if value is None else f'{100*value:.2f}'

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', type=Path, default=ROOT / 'results')
    parser.add_argument('--figures-dir', type=Path, default=ROOT / 'figures')
    parser.add_argument('--ood-stratum', default='semantic_ood')
    args = parser.parse_args()
    summary = json.loads((args.results_dir / 'summary.json').read_text())
    comparisons = json.loads((args.results_dir / 'comparisons.json').read_text())
    selected = summary['checkpoints']
    rows = ['# Tables from saved measurements', '',
            'Probabilities and rates below are percentages. Arm rows average only the observed repair seeds; base and installed checkpoints each have one measurement. These tables do not establish broad model/domain uncertainty.', '',
            '| Arm | Checkpoints | OOD conditional conceal | OOD seed range | Boundary overreport | Audited validity |',
            '|---|---:|---:|---|---:|---:|']
    aggregates = {}
    for arm in ORDER:
        records = [r for r in selected.values() if r['arm'] == arm]
        if not records:
            continue
        ood = [r['strata'][args.ood_stratum]['eliciting_conditional_failure'] for r in records]
        boundary = [r['strata']['all']['boundary_conditional_overreport'] for r in records]
        validity = [r['strata']['all']['generation_valid_rate'] for r in records]
        value = {'n_checkpoints': len(records), 'ood_conditional_conceal_mean': statistics.mean(ood),
                 'ood_conditional_conceal_min': min(ood), 'ood_conditional_conceal_max': max(ood),
                 'boundary_overreport_mean': statistics.mean(boundary), 'generation_validity_mean': statistics.mean(validity)}
        aggregates[arm] = value
        rows.append(f"| {LABELS[arm]} | {len(records)} | {percentage(value['ood_conditional_conceal_mean'])} | "
                    f"{percentage(min(ood))}–{percentage(max(ood))} | {percentage(value['boundary_overreport_mean'])} | "
                    f"{percentage(value['generation_validity_mean'])} |")
    rows += ['', '## Primary contrasts', '',
             'Positive differences favor reactive reflection. Intervals are paired prompt-cluster intervals conditional on the observed models and templates.', '',
             '| Comparator − reactive | Seed | Difference (pp) | 95% prompt-cluster interval (pp) |',
             '|---|---|---:|---|']
    for key, comparison in comparisons['comparisons'].items():
        name = key.removeprefix('reactive_vs_')
        for value in comparison['per_seed'] + ([comparison['observed_seed_mean']] if comparison['observed_seed_mean'] else []):
            seed = value.get('repair_seed', 'Observed-seed mean')
            low, high = value['prompt_cluster_bootstrap_ci95']
            rows.append(f"| {name} | {seed} | {100*value['effect_comparator_minus_reactive']:+.2f} | {100*low:+.2f}, {100*high:+.2f} |")
    rows += ['', '## Every checkpoint', '',
             '| Checkpoint | OOD conditional conceal | OOD argmax conceal | Boundary overreport | Audited validity | Target tokens |',
             '|---|---:|---:|---:|---:|---:|']
    for name, record in sorted(selected.items()):
        ood = record['strata'][args.ood_stratum]
        all_rows = record['strata']['all']
        rows.append(f"| {name} | {percentage(ood['eliciting_conditional_failure'])} | {percentage(ood['eliciting_argmax_failure'])} | "
                    f"{percentage(all_rows['boundary_conditional_overreport'])} | {percentage(all_rows['generation_valid_rate'])} | "
                    f"{record['training'].get('total_target_tokens') or '—'} |")
    rows += ['', '## Free-generation audit', '',
             'Each denominator includes invalid responses. A lower parsed CONCEAL rate is not a success if output validity collapses. '
             'The main audit uses the first 16 cases per stratum, so these rows pool its audited eliciting cases across strata; '
             'they are not estimates of the 72-case primary endpoint. The stress audit covers every stress case.', '',
             '| Checkpoint | All audited | Correct / all audited (%) | Eliciting audited | Parsed CONCEAL (%) | Invalid eliciting (%) |',
             '|---|---:|---:|---:|---:|---:|']
    for name, record in sorted(selected.items()):
        audit = record['strata']['all']
        rows.append(f"| {name} | {audit['n_generation_audited']} | {percentage(audit['generation_correct_rate_all_audited'])} | "
                    f"{audit['n_eliciting_generation_audited']} | {percentage(audit['generation_eliciting_failure_rate_all_audited'])} | "
                    f"{percentage(audit['generation_eliciting_invalid_rate'])} |")
    (args.results_dir / 'report_tables.md').write_text('\n'.join(rows) + '\n')
    (args.results_dir / 'report_values.json').write_text(json.dumps({
        'created_utc': datetime.now(timezone.utc).isoformat(), 'ood_stratum': args.ood_stratum,
        'source_provenance': summary['provenance'], 'arms': aggregates}, indent=2) + '\n')
    plot_contrasts(comparisons, args.figures_dir)
    plot_domains(selected, args.ood_stratum, args.figures_dir)
    print(args.results_dir / 'report_tables.md')

def plot_contrasts(document, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True, sharey=True)
    for ax, (key, comparison) in zip(axes, document['comparisons'].items()):
        values = comparison['per_seed'] + ([comparison['observed_seed_mean']] if comparison['observed_seed_mean'] else [])
        if not values:
            ax.text(.5, .5, 'No paired measurements', ha='center', transform=ax.transAxes)
            continue
        positions = list(range(len(values)))
        for i, value in enumerate(values):
            point = 100*value['effect_comparator_minus_reactive']
            low, high = [100*x for x in value['prompt_cluster_bootstrap_ci95']]
            ax.hlines(i, low, high, color='#2166ac', linewidth=2)
            ax.plot(point, i, 'D' if 'repair_seed' not in value else 'o', color='#2166ac', markersize=7)
        ax.set_yticks(positions, [f"Seed {v['repair_seed']}" if 'repair_seed' in v else 'Observed-seed mean' for v in values])
        ax.axvline(0, color='black', linewidth=1)
        ax.axvline(5, color='#999999', linestyle=':', linewidth=1)
        ax.set_title(key.removeprefix('reactive_vs_').capitalize() + ' minus reactive')
        ax.set_xlabel('Difference in conditional concealment (pp)\nPositive favors reactive')
        ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis()
    fig.suptitle('Paired effects on ' + ', '.join(document['ood_strata']).replace('_', ' '))
    fig.tight_layout()
    fig.text(.02, -.07, 'Bars: 95% paired prompt-cluster intervals, conditional on observed models/templates.\n'
             'Dotted line: predeclared +5 pp practical reference. These are not uncertainty intervals over new installations.', fontsize=8)
    folder.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'svg']:
        fig.savefig(folder / f'primary_contrasts.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

def plot_domains(checkpoints, stratum, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    domains = sorted({row['value'] for checkpoint in checkpoints.values() for row in checkpoint['factor_slices']
                      if row['stratum'] == stratum and row['factor'] == 'domain'})
    arms = [arm for arm in ORDER if any(record['arm'] == arm for record in checkpoints.values())]
    if not domains:
        return
    values, counts = [], set()
    for arm in arms:
        row_values = []
        for domain in domains:
            slices = [row for checkpoint in checkpoints.values() if checkpoint['arm'] == arm
                      for row in checkpoint['factor_slices']
                      if row['stratum'] == stratum and row['factor'] == 'domain' and row['value'] == domain]
            counts.update(row['n_eliciting_report'] for row in slices)
            row_values.append(statistics.mean(row['eliciting_conditional_failure'] for row in slices))
        values.append(row_values)
    values = np.asarray(values)
    fig, ax = plt.subplots(figsize=(max(8, len(domains)*1.4), 5.4))
    im = ax.imshow(100*values, vmin=0, vmax=100, cmap='magma', aspect='auto')
    for i in range(len(arms)):
        for j in range(len(domains)):
            ax.text(j, i, f'{100*values[i,j]:.1f}', ha='center', va='center',
                    color='white' if values[i,j] < .55 else 'black', fontsize=10)
    ax.set_xticks(range(len(domains)), [domain.replace('_', '\n') for domain in domains], fontsize=9)
    ax.set_yticks(range(len(arms)), [LABELS[arm] for arm in arms])
    ax.set_title('Conditional concealment by authored domain template')
    fig.colorbar(im, ax=ax, label='Conditional CONCEAL probability (%)')
    fig.text(.02, -.02, 'Repair rows average observed seeds; base and installed checkpoint each have one measurement.\n'
             f'Eliciting REPORT-required cases per domain/model: {", ".join(map(str, sorted(counts)))}. '
             'These domains are authored templates, not replicated populations.', fontsize=8)
    fig.tight_layout()
    for extension in ['png', 'svg']:
        fig.savefig(folder/f'domains.{extension}', dpi=180, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
