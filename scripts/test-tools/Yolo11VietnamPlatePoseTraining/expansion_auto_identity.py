"""Owner-authorized, audited auto-confirmation of clear validation identities.

Only identity metadata may change. Membership, source bytes, corner labels and
training approvals are never changed by this worker.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from expansion_triage import (OCR_SCHEMA, triage_plate, triage_valid, triage_score,
                              triage_matches, triage_image_matches,
                              triage_validation_index)

METHOD = 'auto-ocr-v1'
MIN_SCORE = .95


def auto_identity_evaluate(row, index):
    p, e = json.loads(row['proposal']), json.loads(row['edit'])
    o = p.get('ocr', {})
    best = triage_plate(o.get('plate', ''))
    reason = None
    if row['kind'] != 'validation' or row['stage'] != 'ready' or e.get('exclude') or p.get('source_deleted'):
        reason = 'not-an-eligible-validation-image'
    elif e.get('identity_reviewed') and e.get('identity_method') != METHOD:
        reason = 'human-confirmed-preserved'
    elif (o.get('schema') != OCR_SCHEMA or not o.get('layout_aware') or
          o.get('crop_pipeline') != 1 or o.get('crop_check', {}).get('status') != 'pass'):
        reason = 'waiting-for-crop-preflight'
    elif o.get('error') or o.get('full_frame') or o.get('regions_truncated') or o.get('region_count') != 1:
        reason = 'uncertain-or-multiple-regions'
    elif not triage_valid(best) or triage_score(o.get('confidence')) < MIN_SCORE:
        reason = 'low-confidence-or-invalid-plate'
    elif e.get('identity_method') != METHOD and e.get('plate') and triage_plate(e['plate']) != best:
        reason = 'saved-manual-plate-differs'
    methods = {c.get('method') for c in o.get('candidates', [])
               if triage_plate(c.get('plate', '')) == best
               and c.get('region', 0) == o.get('region', 0)
               and triage_score(c.get('confidence')) >= MIN_SCORE}
    if not reason and not ('det-lines' in methods and methods & {'whole-crop', 'split-lines'}):
        reason = 'two-high-confidence-methods-required'
    current_alternatives = {triage_plate(c.get('plate', '')) for c in o.get('candidates', [])
                            if triage_valid(triage_plate(c.get('plate', '')))
                            and triage_score(c.get('confidence')) >= .7}
    historical_alternatives = {triage_plate(c.get('plate', '')) for c in o.get('comparison_candidates', [])
                               if triage_valid(triage_plate(c.get('plate', '')))
                               and triage_score(c.get('confidence')) >= MIN_SCORE}
    if not reason and (current_alternatives | historical_alternatives) - {best}:
        reason = 'conflicting-plate-evidence'
    matches = []
    if not reason:
        evidence = [dict(plate=best, confidence=triage_score(o.get('confidence')))]
        matches = triage_matches(evidence, index) + triage_image_matches(row, p, index, best)
        # Repeated pictures of exactly the same plate belong to one vehicle;
        # different plates one edit apart / near-identical images need review.
        matches = [m for m in matches if m['id'] != row['id'] and m.get('reference_plate') != best]
        if matches: reason = 'ambiguous-validation-reference'
    digest = hashlib.sha256(json.dumps(dict(ocr=o, sha=row['sha'], policy=METHOD),
        sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    return dict(eligible=reason is None, reason=reason, plate=best,
                methods=sorted(methods), evidence_digest=digest,
                minimum_confidence=MIN_SCORE, reference_ids=sorted({m['id'] for m in matches}))


def auto_identity_pass(workspace):
    from curate_acs_training_pool import expansion_db, expansion_meta
    counts = Counter()
    with expansion_db(workspace) as db:
        if not expansion_meta(db, 'manifest_sha256'):
            raise ValueError('Initialize the frozen validation workspace first')
        db.execute('CREATE TABLE IF NOT EXISTS auto_identity_audit('
                   'id TEXT,version INTEGER,action TEXT,evidence TEXT,at INTEGER,PRIMARY KEY(id,version))')
        db.execute('CREATE TABLE IF NOT EXISTS crop_ocr_priority(id TEXT PRIMARY KEY,priority INTEGER NOT NULL)')
        db.commit()
        db.execute('BEGIN IMMEDIATE')
        rows = db.execute("SELECT * FROM images WHERE kind='validation'").fetchall()
        index = triage_validation_index(rows)
        for row in rows:
            p, e = json.loads(row['proposal']), json.loads(row['edit'])
            if e.get('identity_reviewed') and e.get('identity_method') != METHOD:
                counts['human_confirmed'] += 1;continue
            decision = auto_identity_evaluate(row, index)
            was_auto = bool(e.get('identity_reviewed') and e.get('identity_method') == METHOD)
            if decision['eligible']:
                if was_auto and e.get('identity_evidence', {}).get('evidence_digest') == decision['evidence_digest']:
                    counts['auto_confirmed'] += 1;continue
                updated = dict(e, plate=decision['plate'], identity_reviewed=True,
                    reviewer='OCR tự động', identity_method=METHOD, identity_evidence=decision)
                action = 'confirm'
            elif was_auto:
                updated = dict(e, identity_reviewed=False, identity_method=METHOD,
                               reviewer='OCR tự động — cần kiểm tra', identity_evidence=decision)
                updated.pop('plate', None)
                action = 'revoke'
            else:
                counts['needs_review'] += 1
                # Keep reviewer edits and versions untouched when merely placing
                # an uncertain record in the manual queue.
                hold = dict(status='needs_review', reason=decision['reason'],
                            reference_ids=decision['reference_ids'])
                if p.get('auto_identity_review') != hold:
                    p['auto_identity_review'] = hold
                    db.execute('UPDATE images SET proposal=? WHERE id=?', (json.dumps(p), row['id']))
                if p.get('ocr', {}).get('crop_pipeline') != 1:
                    old=p.get('ocr',{})
                    support={c.get('method') for c in old.get('candidates',[])
                             if c.get('plate')==old.get('plate') and triage_score(c.get('confidence'))>=MIN_SCORE}
                    priority=-1 if 'det-lines' in support and support & {'split-lines','whole-crop'} else 0
                    # Prioritize promising old reads for REAL crop preflight;
                    # old agreement alone is never confirmation evidence.
                    db.execute('INSERT INTO crop_ocr_priority VALUES(?,?) ON CONFLICT(id) DO UPDATE SET priority=excluded.priority', (row['id'],priority))
                    counts['waiting_crop'] += 1
                continue
            now = time.time_ns()
            db.execute('INSERT INTO history VALUES(?,?,?,?)', (row['id'], row['version'], row['edit'], now))
            db.execute('INSERT INTO auto_identity_audit VALUES(?,?,?,?,?)',
                       (row['id'], row['version']+1, action, json.dumps(decision), now))
            p['auto_identity_review'] = dict(status='confirmed' if action == 'confirm' else 'needs_review',
                reason=decision['reason'], reference_ids=decision['reference_ids'])
            db.execute('UPDATE images SET edit=?,proposal=?,version=version+1 WHERE id=? AND version=?',
                       (json.dumps(updated), json.dumps(p), row['id'], row['version']))
            counts['newly_confirmed' if action == 'confirm' else 'revoked'] += 1
            counts['auto_confirmed' if action == 'confirm' else 'needs_review'] += 1
        db.commit()
        counts['total'] = len(rows)
        result = dict(counts, at=time.time(), policy=METHOD, minimum_confidence=MIN_SCORE)
        expansion_meta(db, 'auto_identity_progress', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--follow', action='store_true')
    args = parser.parse_args()
    while True:
        result = auto_identity_pass(args.workspace)
        print(json.dumps(result), flush=True)
        output = Path(args.workspace)/'auto-identity-progress.json'
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result), encoding='utf-8')
        temporary.replace(output)
        if not args.follow or not result.get('waiting_crop'): break
        time.sleep(30)
