"""Small read-only progress display; partial rows are never scientific results."""
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'


def rows(path):
    if not path.exists():
        return []
    output = []
    for line in path.read_text(errors='replace').splitlines():
        try:
            output.append(json.loads(line))
        except ValueError:
            continue
    return output


def main():
    progress_path = RESULTS / 'PROGRESS.json'
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    result = dict(progress=progress, completed_stages=len(list((RESULTS / 'stages').glob('*/COMPLETED.json'))))
    stage = progress.get('stage')
    if stage:
        directory = RESULTS / 'stages' / stage
        outputs = rows(directory / 'outputs.jsonl')
        training = rows(directory / 'training.jsonl')
        result['partial_outputs'] = len(outputs)
        result['partial_loss_rows'] = len(rows(directory / 'target_losses.jsonl'))
        if training:
            result['last_training_step'] = {key: training[-1][key] for key in ('step', 'loss', 'elapsed_s')}
        if (directory / 'FAILED.json').exists():
            result['stage_failure'] = json.loads((directory / 'FAILED.json').read_text())
    allocation = RESULTS / 'allocation.json'
    if allocation.exists():
        state = json.loads(allocation.read_text())
        result['allocated_minutes'] = round((time.time() - state['started_unix']) / 60, 2)
    terminal = RESULTS / 'TERMINAL.json'
    if terminal.exists():
        result['terminal_status'] = json.loads(terminal.read_text())['status']
    print(json.dumps(result))


if __name__ == '__main__':
    main()
