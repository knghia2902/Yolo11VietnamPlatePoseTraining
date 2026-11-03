"""Read-only visual-review overlay; never grants training or human approval."""
import json
from pathlib import Path


def load_corner_review_index(workspace):
    try:
        payload = json.loads((Path(workspace)/'corner-review-index.json').read_text(encoding='utf-8'))
        return payload['records'] if isinstance(payload, dict) and payload.get('schema') == 1 and isinstance(payload.get('records'), dict) else {}
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def needs_corner_review(row, index):
    proposal, edit = json.loads(row['proposal']), json.loads(row['edit'])
    if row['kind'] not in ('acs', 'roboflow') or edit.get('corners_reviewed'):
        return False
    if edit.get('exclude') or proposal.get('source_deleted') or row['stage'] in ('excluded', 'duplicate'):
        return False
    entry = index.get(row['id'])
    if not isinstance(entry, dict):
        return True
    return not (entry.get('status') == 'checked_visual'
                and entry.get('version') == row['version']
                and entry.get('sha256') == row['sha']
                and entry.get('points') == edit.get('points', proposal.get('points')))
