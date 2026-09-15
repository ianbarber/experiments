#!/usr/bin/env python3
"""Build a NEW continuation draft after terminal and verified service restoration.

The destination contains only code/calibration and results/calibration. It is not
an existing entry or Git checkout. No publishing, editorial edits or model calls.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import posixpath
import re
import shutil
from datetime import datetime, timezone
from urllib.parse import quote, unquote, urlsplit

HERE=Path(__file__).resolve().parent
CODE='code/calibration'
EVIDENCE='results/calibration'
ROW_ORIGIN='_original_record_sha256'
PRIVATE_KEYS={'service_id','research_id','container_id','pid','session_pid','hostname','host','machine_serial',
              'token_ids','input_ids','labels','optimizer_state_dict','text_with_special_tokens','trainable_parameter_names'}
BAD_PUBLIC=re.compile(r'(?:sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}|AKIA[A-Z0-9]{12,}|PRIVATE[ ]KEY|\.ts\.net|\b(?:192\.168|10\.[0-9]+)\.[0-9]+\.[0-9]+|/home/[A-Za-z0-9_.-]+(?:/|\b))')
SUMMARY_FILES=('TERMINAL.json','allocation.json','allocation_released.json','service_restoration.json',
               'INITIAL_GRID.json','ACQUISITION_CONTROL.json','SHORTLIST_LOCK.json','REPLICATIONS.json',
               'PAIRING_DEVELOPMENT.json','CONFIRMATION_LOCK.json','CONFIRMATIONS.json','MATERIALS.json','CONTENT_RESULTS.json')
STAGE_JSON={'started.json','COMPLETED.json','FAILED.json','candidate.json','recomputed_scores.json',
            'acquisition_summary.json','target_loss_summary.json','generation_batches.json','adapter/adapter_config.json'}
STAGE_ROWS={'outputs.jsonl','target_losses.jsonl','exposure.jsonl','training.jsonl'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def row_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+'\n')


class Projection:
    def __init__(self,source):
        self.source=Path(source).resolve()
        self.replacements={}
        allocation=self.source/'results/allocation.json'
        if allocation.exists():
            value=read(allocation)
            for key,alias in [('service_id','original-serving-container'),('research_id','isolated-research-container')]:
                if value.get(key): self.replacements[value[key]]=alias
        config=self.source/'configs/pilot.json'
        if config.exists():
            value=read(config)
            for key,alias in [('service_container','original-serving-container'),('research_container','isolated-research-container')]:
                if value.get(key): self.replacements[value[key]]=alias

    def text(self,text,python=False):
        for private,alias in self.replacements.items(): text=text.replace(private,alias)
        text=text.replace(str(self.source)+'/', '').replace(str(self.source),'calibration-workspace')
        if not python: text=text.replace('/workspace/','').replace('/workspace','calibration-workspace')
        text=re.sub(r'/home/[A-Za-z0-9_.-]+(?:/[^\s"\x27<>)]*)?', '[private-path-removed]',text)
        return text

    def value(self,value):
        if isinstance(value,dict):
            for key in ('text','generated_text','before','after','content'):
                if isinstance(value.get(key),str) and self.text(value[key])!=value[key]:
                    raise ValueError('Privacy projection would alter protected scientific text: '+key)
            result={}
            for key,item in value.items():
                if key in PRIVATE_KEYS: continue
                if key=='unexpected_special_token_ids': result['unexpected_special_token']=bool(item)
                else: result[self.text(key)]=self.value(item)
            return result
        if isinstance(value,list): return [self.value(item) for item in value]
        return self.text(value) if isinstance(value,str) else value


def project_analysis_script_links(text,source,original_path,public_path,records):
    """Relocate inline links to already-retained analysis scripts, declaring each.

    Other links and prose remain unchanged. This intentionally supports the
    simple inline Markdown links in authored supporting reviews, not a general
    Markdown rewrite or a second copy of the referenced code.
    """
    indexed={record['original_path']:record for record in records}; projected=[]
    pattern=re.compile(r'(\[[^\]\n]+\]\()(<[^>\n]+>|[^\s()]+)((?:\s+"[^"\n]*")?\))')
    def replace(match):
        destination=match[2]; angled=destination.startswith('<'); href=destination[1:-1] if angled else destination
        parts=urlsplit(href)
        if parts.scheme or parts.netloc or not parts.path or parts.path.startswith('/'): return match[0]
        original=posixpath.normpath(posixpath.join(posixpath.dirname(original_path),unquote(parts.path)))
        if not original.startswith('results/analysis_tools/') or not original.endswith('.py'): return match[0]
        target=source/original; resolved=target.resolve()
        if not resolved.is_relative_to(source/'results/analysis_tools') or not resolved.is_file():
            raise ValueError('Supporting review analysis-script link is missing or escapes its source directory: '+original)
        mapping=indexed.get(original)
        if (mapping is None or mapping['public_path']!=CODE+'/'+original
                or mapping['original_sha256']!=sha(target)):
            raise ValueError('Supporting review analysis-script link lacks an unchanged retained transport target: '+original)
        relative=posixpath.relpath(mapping['public_path'],posixpath.dirname(public_path))
        replacement=quote(relative,safe='/._-~')+('?' + parts.query if parts.query else '')+('#'+parts.fragment if parts.fragment else '')
        projected.append(dict(original_destination=href,public_destination=replacement,target_original_path=original,
                         target_original_sha256=mapping['original_sha256'],target_public_path=mapping['public_path'],
                         target_public_sha256=mapping['public_sha256']))
        return match[1]+('<'+replacement+'>' if angled else replacement)+match[3]
    return pattern.sub(replace,text),projected


def read_complete_lines(path,allow_partial=False):
    raw=Path(path).read_bytes(); result=[]; omitted=None
    lines=raw.splitlines(keepends=True)
    for index,line in enumerate(lines):
        if not line.strip(): continue
        try: result.append(json.loads(line))
        except (ValueError,UnicodeDecodeError):
            if not allow_partial or index!=len(lines)-1: raise ValueError('Malformed JSONL: '+str(path))
            omitted=dict(bytes=len(line),sha256=hashlib.sha256(line).hexdigest(),reason='Unterminated final record in an incomplete stage; not numerical evidence.')
    return result,omitted


def audit_closed(source):
    source=Path(source)
    terminal=read(source/'results/TERMINAL.json')
    restored=read(source/'results/service_restoration.json')
    allocation=read(source/'results/allocation.json'); released=read(source/'results/allocation_released.json')
    if (restored.get('health_status')!=200 or restored.get('research_running') is not False
            or restored.get('same_original_container_and_image') is not True or released.get('research_running') is not False):
        raise ValueError('Export waits for healthy exact-service restoration and confirmed research release.')
    elapsed=released['released_unix']-allocation['started_unix']
    if not math.isfinite(elapsed) or elapsed<0 or elapsed>allocation['budget_seconds']:
        raise ValueError('Invalid or exceeded authorized allocation; resolve explicit reporting before export.')
    if not math.isclose(restored['research_allocation_elapsed_seconds'],elapsed,abs_tol=1e-6):
        raise ValueError('Restoration elapsed time differs from release receipt.')
    if terminal.get('repair_training_launched') is not False:
        raise ValueError('Continuation scope unexpectedly includes repair training.')
    if terminal.get('freeze_sha256') not in (None,sha(source/'FREEZE.json')):
        raise ValueError('Terminal refers to a different scientific freeze.')
    return terminal


def export(source,destination,run_replay=True,supporting=()):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError('Use a new draft directory outside the source experiment.')
    if any((parent/'.git').exists() for parent in (destination,*destination.parents)):
        raise ValueError('Draft export must not write inside a Git checkout.')
    terminal=audit_closed(source)
    frozen=read(source/'FREEZE.json')
    for name,digest in frozen['files'].items():
        if sha(source/name)!=digest: raise ValueError('Frozen source changed: '+name)
    projection=Projection(source)
    destination.mkdir(parents=True)
    transport=dict(version=1,created_at=datetime.now(timezone.utc).isoformat(),original_freeze_sha256=sha(source/'FREEZE.json'),
        scope='Continuation only. Exact decoded generated text and scalar records are retained; token arrays, text with control tokens, weights, optimizer state, raw operational logs, agent sessions, host/process/container identities and raw regenerated datasets are omitted. Original and public hashes bind an explicitly lossy projection; omitted bytes cannot be reconstructed from this public archive.',
        original_row_hash_scope='Each projected JSONL record carries the canonical hash of its original full record. These preserved assertions keep reflection-source links checkable; they do not reconstruct or independently authenticate omitted token arrays.',
        frozen_sources={},records=[],stages=[],editorial_scope='Existing README, REPORT, LABNOTES and figures are authored/merged separately; this draft does not modify them.')

    def capture(path,public=None,mode=None,partial=False,analysis_links=False):
        relative=str(path.relative_to(source)); public=public or EVIDENCE+'/'+relative.removeprefix('results/')
        target=destination/public; target.parent.mkdir(parents=True,exist_ok=True)
        record=dict(original_path=relative,original_sha256=sha(path),public_path=public)
        if path.suffix=='.jsonl':
            records,omitted=read_complete_lines(path,partial)
            projected=[]
            for row in records:
                if ROW_ORIGIN in row: raise ValueError('Reserved projection metadata already present.')
                projected.append(dict(projection.value(row),**{ROW_ORIGIN:row_sha(row)}))
            raw=''.join(json.dumps(row,sort_keys=True,ensure_ascii=False)+'\n' for row in projected).encode()
            public += '.gz'; target=destination/public
            target.write_bytes(gzip.compress(raw,compresslevel=9,mtime=0))
            record.update(public_path=public,mode=mode or 'gzip_text_scalar_projection',rows=len(projected))
            if omitted: record['omitted_trailing_fragment']=omitted
        elif path.suffix=='.json':
            value=projection.value(read(path)); write(target,value)
            record['mode']=mode or ('byte_identical' if target.read_bytes()==path.read_bytes() else 'json_projection')
        else:
            raw=path.read_bytes(); transformed=projection.text(raw.decode(),python=path.suffix=='.py')
            if analysis_links and path.suffix=='.md':
                transformed,links=project_analysis_script_links(transformed,source,relative,public,transport['records'])
                if links:
                    record['analysis_script_link_projection']=links
                    mode='sanitized_text_with_analysis_script_link_projection'
            transformed=transformed.encode()
            target.write_bytes(transformed)
            record['mode']=mode or ('byte_identical' if raw==transformed else 'sanitized_text')
        record['public_sha256']=sha(target)
        transport['records'].append(record)
        return record

    freeze_public=destination/CODE/'FREEZE.json'; freeze_public.parent.mkdir(parents=True)
    if projection.text((source/'FREEZE.json').read_text())!=(source/'FREEZE.json').read_text():
        raise ValueError('Freeze contains private text; cannot claim exact freeze transport.')
    shutil.copyfile(source/'FREEZE.json',freeze_public)
    for name,digest in frozen['files'].items():
        path=source/name
        if name.startswith('data/') and path.suffix=='.jsonl':
            mapping=dict(original_sha256=digest,mode='regenerated',public_path=None)
        elif path.suffix in ('.safetensors','.pt','.bin'):
            mapping=dict(original_sha256=digest,mode='omitted_checkpoint',public_path=None)
        else:
            record=capture(path,CODE+'/'+name)
            mapping={key:record[key] for key in ('original_sha256','mode','public_path','public_sha256')}
        transport['frozen_sources'][name]=mapping
    stage_root=source/'results/stages'
    for stage in sorted(stage_root.iterdir() if stage_root.exists() else []):
        if not stage.is_dir(): continue
        completed=read(stage/'COMPLETED.json') if (stage/'COMPLETED.json').exists() else None
        started=read(stage/'started.json') if (stage/'started.json').exists() else {}
        actual=bool(completed and completed.get('status')=='complete' and completed.get('check_only') is False)
        if completed:
            for artifact,digest in completed['artifacts_sha256'].items():
                if sha(stage/artifact)!=digest: raise ValueError('Changed completed-stage artifact: '+stage.name+'/'+artifact)
        item=dict(name=stage.name,mode=started.get('mode',completed.get('mode') if completed else None),
                  status='complete' if actual else 'check_only' if completed else 'incomplete',files={},omitted_artifacts={})
        for name in sorted(STAGE_JSON|STAGE_ROWS):
            path=stage/name
            if path.exists():
                record=capture(path,partial=not actual)
                item['files'][name]=record['public_path']
        if completed:
            item['omitted_artifacts']={name:digest for name,digest in completed['artifacts_sha256'].items() if name not in item['files']}
        transport['stages'].append(item)
    for name in SUMMARY_FILES:
        path=source/'results'/name
        if path.exists(): capture(path)
    for directory in ('gates','derived','content_review'):
        for path in sorted((source/'results'/directory).rglob('*')) if (source/'results'/directory).exists() else []:
            if path.is_file() and path.suffix in ('.json','.jsonl','.md'): capture(path)
    # Preserve helper locations: their parents[2] root remains code/calibration.
    for path in sorted((source/'results/analysis_tools').glob('*')):
        if path.suffix=='.py': capture(path,CODE+'/results/analysis_tools/'+path.name)
    # Final narrative/audit snapshots are selected explicitly, not live watcher history.
    for relative in supporting:
        path=(source/relative).resolve()
        if not path.is_relative_to(source) or path.suffix not in ('.json','.md','.csv'):
            raise ValueError('Supporting files must be source-relative JSON, Markdown or CSV.')
        if any(record['original_path']==str(path.relative_to(source)) for record in transport['records']):
            continue
        capture(path,EVIDENCE+'/supporting/'+str(path.relative_to(source)),analysis_links=True)
    for name in ('export.py','replay.py','test_publication.py','test_material_paths.py'):
        path=HERE/name
        if path.is_file(): capture(path,CODE+'/'+name,'publication_helper_snapshot')
    instructions=(HERE/'README.md').read_text()
    (destination/CODE/'README.md').write_text(instructions)
    write(destination/EVIDENCE/'STAGES.json',transport['stages'])
    write(destination/EVIDENCE/'EXPORT_STATUS.json',dict(status='new_continuation_draft',terminal_status=terminal['status'],
          content_result_present=(source/'results/CONTENT_RESULTS.json').exists(),repair_training_launched=False,
          note='Numerical replay is evidence replay, not fresh model replication. Editorial materials and figures are supplied separately.'))
    for path in destination.rglob('*'):
        if path.is_file():
            raw=gzip.decompress(path.read_bytes()).decode() if path.suffix=='.gz' else path.read_text()
            if BAD_PUBLIC.search(raw): raise ValueError('Public privacy scan failed: '+str(path.relative_to(destination)))
    transport['public_artifacts']={str(path.relative_to(destination)):sha(path) for path in sorted(destination.rglob('*')) if path.is_file()}
    write(destination/EVIDENCE/'TRANSPORT.json',transport)
    result=dict(destination=str(destination),terminal_status=terminal['status'],stages=len(transport['stages']),
                completed_stages=sum(item['status']=='complete' for item in transport['stages']))
    if run_replay:
        import importlib.util
        spec=importlib.util.spec_from_file_location('calibration_publication_replay',HERE/'replay.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        numerical=module.replay(destination)
        write(destination/EVIDENCE/'NUMERICAL_REPLAY.json',numerical)
        result['numerical_replay']=numerical
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--supporting',action='append',default=[],help='Explicit final audit/analysis/note path relative to source; repeat as needed.')
    args=parser.parse_args()
    print(json.dumps(export(args.source,args.destination,supporting=args.supporting),indent=2))
