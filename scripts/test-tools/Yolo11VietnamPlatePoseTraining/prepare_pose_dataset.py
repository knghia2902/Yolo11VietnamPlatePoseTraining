"""Offline single-plate annotation, deduplication and group-held-out packaging.

Only human-reviewed corners may enter a training package. No network requests.
Original archives are read-only; generated images/annotations stay outside Git.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import zipfile

from PIL import Image, ImageOps

ORDER = ["TL", "TR", "BR", "BL"]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def fresh_output(path):
    path = Path(path).resolve()
    repo = Path(__file__).resolve().parents[3]
    if path == repo or repo in path.parents:
        raise ValueError("Generated datasets must be outside the repository")
    path.mkdir(parents=True, exist_ok=False)
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fingerprints(data):
    with Image.open(io.BytesIO(data)) as original:
        if original.getexif().get(274, 1) != 1:
            raise ValueError("EXIF rotation requires explicit coordinate conversion")
        image = original.convert("RGB")
        image.load()
    thumb = list(ImageOps.grayscale(image).resize((9, 8)).getdata())
    dhash = sum((thumb[y * 9 + x] > thumb[y * 9 + x + 1]) << (y * 8 + x)
                for y in range(8) for x in range(8))
    pixel_hash = digest(str(image.size).encode() + image.tobytes())
    return image.size, pixel_hash, dhash


def valid_points(points):
    if len(points) != 4 or any(len(p) != 2 for p in points):
        raise ValueError("Exactly four [x,y] points required")
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
           for p in points for v in p):
        raise ValueError("Corners must be finite normalized coordinates")
    cross = []
    for i in range(4):
        a, b, c = points[i], points[(i + 1) % 4], points[(i + 2) % 4]
        cross.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if min(cross) <= 1e-8:
        raise ValueError("Corners must form a nondegenerate clockwise convex polygon")
    # The dataset contract excludes upside-down plates: TL/TR precede BL/BR.
    if points[0][1] + points[1][1] >= points[2][1] + points[3][1]:
        raise ValueError("Top and bottom corners are swapped")
    if points[0][0] + points[3][0] >= points[1][0] + points[2][0]:
        raise ValueError("Left and right corners are swapped")


def parse_label(text):
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("This review tool requires one fully annotated plate per image; multi-plate images need a multi-object annotation tool")
    a = list(map(float, lines[0].split()))
    if len(a) != 17 or not all(math.isfinite(v) for v in a) or a[0] != 0:
        raise ValueError("Expected one class-0 YOLO pose row with 17 finite values")
    if any(v not in (0, 1, 2) for v in a[7::3]):
        raise ValueError("Invalid visibility")
    if not all(0 <= v <= 1 for v in a[1:5]) or min(a[3:5]) <= 0:
        raise ValueError("Invalid bounding box")
    points = [[a[i], a[i + 1]] for i in (5, 8, 11, 14)]
    valid_points(points)
    return a[1:5], points


def contour_proposal(data, bbox):
    """Heuristic only: never assigns review approval or overwrites source labels."""
    import cv2
    import numpy as np
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    x0, y0 = max(0, int((x - bw * .6) * w)), max(0, int((y - bh * .6) * h))
    x1, y1 = min(w, int((x + bw * .6) * w)), min(h, int((y + bh * .6) * h))
    if x1 <= x0 or y1 <= y0:
        return None
    crop = image[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if not .2 * bw * w * bh * h <= area <= 1.4 * bw * w * bh * h:
            continue
        polygon = cv2.approxPolyDP(contour, .02 * cv2.arcLength(contour, True), True)
        if len(polygon) != 4 or not cv2.isContourConvex(polygon):
            continue
        pts = polygon.reshape(4, 2).astype(float)
        center = pts.mean(axis=0)
        pts = pts[np.argsort(np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0]))]
        pts = np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0)
        proposal = [[float((p[0] + x0) / w), float((p[1] + y0) / h)] for p in pts]
        try:
            valid_points(proposal)
            return proposal
        except ValueError:
            continue
    return None


def scan(source, out, propose=False):
    out = fresh_output(out)
    (out / "images").mkdir()
    records, seen, errors = [], {}, []
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member names")
        images = sorted(n for n in names if n.startswith("images/") and PurePosixPath(n).suffix.lower() in (".jpg", ".jpeg", ".png"))
        for name in images:
            try:
                if len(PurePosixPath(name).parts) != 2 or "\\" in name:
                    raise ValueError("Expected flat images/ directory")
                data = archive.read(name)
                size, pixel, dhash = fingerprints(data)
                label = "labels/" + PurePosixPath(name).stem + ".txt"
                bbox, points = parse_label(archive.read(label).decode("utf-8-sig"))
                source_group = PurePosixPath(name).stem.split("_")[0]
                if pixel in seen:
                    record = seen[pixel]
                    record["aliases"].append(name)
                    record["source_groups"] = sorted(set(record["source_groups"] + [source_group]))
                    continue
                image_id = digest(data)
                target = image_id + PurePosixPath(name).suffix.lower()
                (out / "images" / target).write_bytes(data)
                proposal = contour_proposal(data, bbox) if propose else None
                record = dict(id=image_id, image="images/" + target, sha256=image_id,
                              pixel_sha256=pixel, dhash=dhash, size=list(size), aliases=[name],
                              source_groups=[source_group], group=source_group,
                              group_reviewed=False, reviewed=False, exclude=False,
                              proposal="contour_unreviewed" if proposal else "legacy_bbox_unreviewed",
                              points=proposal or points)
                records.append(record)
                seen[pixel] = record
            except (ValueError, KeyError, OSError) as exc:
                errors.append({"image": name, "error": str(exc)})
    manifest = dict(schema=1, corner_order=ORDER, source_sha256=digest(Path(source).read_bytes()),
                    original_images=len(images), unique_images=len(records),
                    removed_duplicates=sum(len(r["aliases"]) - 1 for r in records),
                    errors=errors, records=records)
    write_json(out / "manifest.json", manifest)
    draft_plan(out)
    template = Path(__file__).with_name("review_pose.html").read_text(encoding="utf-8")
    payload = json.dumps(manifest, ensure_ascii=True).replace("<", "\\u003c")
    (out / "review.html").write_text(template.replace("/*MANIFEST*/null", payload), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=2))
    return manifest


def split_groups(records, fraction=.15, seed=42, distance=4):
    """Union session/source groups and near-duplicate edges before splitting.

    dHash is a conservative screening heuristic, not proof of visual uniqueness.
    Exact duplicates have already been removed by decoded RGB hash.
    """
    if not 0 < fraction < 1 or not 0 <= distance <= 16:
        raise ValueError("Invalid split fraction or dHash threshold")
    parent = list(range(len(records)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[root(j)] = root(i)

    owners = {}
    for i, record in enumerate(records):
        # A reviewer may refine broad source groups to real session/vehicle IDs.
        # Exact duplicates spanning sources connect their original sources too.
        keys = ["group:" + record["group"]]
        if len(record["source_groups"]) > 1:
            keys += ["source:" + s for s in record["source_groups"]]
        keys += ["source:" + s for s in record["source_groups"] if record["group"] == s]
        for key in keys:
            if key in owners:
                union(i, owners[key])
            owners[key] = i
    for i, a in enumerate(records):
        for j in range(i):
            b = records[j]
            if (a["dhash"] ^ b["dhash"]).bit_count() <= distance:
                union(i, j)
    groups = {}
    for i in range(len(records)):
        groups.setdefault(root(i), []).append(i)
    components = sorted(groups.values(), key=lambda group: min(records[i]["id"] for i in group))
    if len(components) < 2:
        raise ValueError("Fewer than two independent groups remain; collect an independent validation set")
    random.Random(seed).shuffle(components)
    # Deterministic subset-sum: nearest achievable ratio without cutting a group.
    possible = {0: []}
    for index, component in enumerate(components):
        for count, chosen in list(possible.items()):
            possible.setdefault(count + len(component), chosen + [index])
    count = min((n for n in possible if 0 < n < len(records)),
                key=lambda n: (abs(n - fraction * len(records)), n))
    val_components = set(possible[count])
    assignment = {}
    for index, component in enumerate(components):
        component_id = min(records[i]["id"] for i in component)
        for i in component:
            assignment[records[i]["id"]] = ("val" if index in val_components else "train", component_id)
    return assignment


def draft_plan(workspace):
    """Preview only; never emits trainable image/label folders."""
    workspace = Path(workspace)
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    plan = dict(training_ready=False, note="Unreviewed source groups and corners; recompute after review")
    try:
        assignment = split_groups(manifest["records"])
        plan["counts"] = {split: sum(v[0] == split for v in assignment.values()) for split in ("train", "val")}
        plan["components"] = len({v[1] for v in assignment.values()})
        plan["assignments"] = {key: {"split": value[0], "component": value[1]} for key, value in assignment.items()}
    except ValueError as exc:
        plan["blocked"] = str(exc)
    write_json(workspace / "draft-split.json", plan)
    return plan


def label_text(points):
    valid_points(points)
    xs, ys = zip(*points)
    box = [(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2,
           max(xs) - min(xs), max(ys) - min(ys)]
    return "0 " + " ".join(f"{v:.8f}" for v in box) + " " + " ".join(
        f"{x:.8f} {y:.8f} 2" for x, y in points) + "\n"


def finalize(workspace, reviews, out, fraction=.15, seed=42, distance=4):
    workspace = Path(workspace)
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    review = json.loads(Path(reviews).read_text(encoding="utf-8"))
    if review.get("source_sha256") != manifest["source_sha256"]:
        raise ValueError("Reviews belong to a different source archive")
    if manifest["errors"]:
        raise ValueError("Scan errors must be resolved before packaging")
    updates = review["records"]
    if len({r["id"] for r in updates}) != len(updates):
        raise ValueError("Duplicate review IDs")
    updates = {r["id"]: r for r in updates}
    if set(updates) != {r["id"] for r in manifest["records"]}:
        raise ValueError("Reviews must cover exactly the scanned image IDs")
    records = []
    for original in manifest["records"]:
        update = updates[original["id"]]
        if update.get("exclude") is True:
            continue
        if update.get("reviewed") is not True or update.get("group_reviewed") is not True:
            raise ValueError("Unreviewed corners or group: " + original["id"])
        group = update.get("group", "").strip()
        if not group or not update.get("reviewer", "").strip():
            raise ValueError("Reviewer and leakage group are required")
        valid_points(update["points"])
        image_path = workspace / original["image"]
        data = image_path.read_bytes()
        size, pixel, dhash = fingerprints(data)
        if digest(data) != original["sha256"] or list(size) != original["size"]:
            raise ValueError("Image changed after review")
        records.append({**original, **{k: update[k] for k in
                       ("points", "reviewed", "group_reviewed", "reviewer")},
                       "group": group, "pixel_sha256": pixel, "dhash": dhash})
    assignment = split_groups(records, fraction, seed, distance)
    out = fresh_output(out)
    rows = []
    for record in records:
        split, component = assignment[record["id"]]
        image = f"images/{split}/{Path(record['image']).name}"
        label = f"labels/{split}/{record['id']}.txt"
        (out / image).parent.mkdir(parents=True, exist_ok=True)
        (out / label).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(workspace / record["image"], out / image)
        text = label_text(record["points"])
        (out / label).write_text(text, encoding="utf-8")
        rows.append({**record, "image": image, "label": label, "split": split,
                     "component": component, "label_sha256": digest(text.encode())})
    package = dict(schema=1, corner_order=ORDER, reviewed=True, source_sha256=manifest["source_sha256"],
                   seed=seed, dhash_distance=distance, requested_val_fraction=fraction,
                   actual_val_fraction=sum(r["split"] == "val" for r in rows) / len(rows),
                   original_images=manifest["original_images"], removed_duplicates=manifest["removed_duplicates"],
                   excluded_images=manifest["unique_images"] - len(rows), records=rows)
    write_json(out / "manifest.json", package)
    (out / "plate_pose.yaml").write_text(
        "# Run from package root; notebook writes an absolute dataset path.\n"
        "train: images/train\nval: images/val\nkpt_shape: [4, 3]\nflip_idx: []\nnames:\n  0: license_plate\n", encoding="utf-8")
    shutil.copyfile(__file__, out / "prepare_pose_dataset.py")
    verify(out)
    if Path(str(out) + "-training.zip").exists():
        raise ValueError("Training archive already exists; choose a new output")
    shutil.make_archive(str(out) + "-training", "zip", out)
    print(json.dumps({k: v for k, v in package.items() if k != "records"}, indent=2))
    return package


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("reviewed") is not True or manifest.get("corner_order") != ORDER:
        raise ValueError("Not a reviewed four-corner training package")
    seen_ids, seen_pixels, groups, hashes = set(), set(), {}, {"train": [], "val": []}
    expected = set()
    for record in manifest["records"]:
        split = record["split"]
        if split not in hashes or record.get("reviewed") is not True or record.get("group_reviewed") is not True:
            raise ValueError("Invalid review or split")
        for key in ("group", "component"):
            group = (key, record[key])
            if group in groups and groups[group] != split:
                raise ValueError("Group leakage")
            groups[group] = split
        for key, directory, suffixes in (("image", "images", (".jpg", ".jpeg", ".png")), ("label", "labels", (".txt",))):
            path = PurePosixPath(record[key])
            if len(path.parts) != 3 or path.parts[:2] != (directory, split) or path.suffix not in suffixes or "\\" in record[key]:
                raise ValueError("Invalid package path")
            expected.add(record[key])
        data = (root / record["image"]).read_bytes()
        size, pixel, dhash = fingerprints(data)
        text = (root / record["label"]).read_text(encoding="utf-8")
        _, points = parse_label(text)
        if record["id"] in seen_ids or pixel in seen_pixels:
            raise ValueError("Duplicate image in package")
        if digest(data) != record["sha256"] or digest(text.encode()) != record["label_sha256"]:
            raise ValueError("Package hash mismatch")
        if text != label_text(record["points"]) or list(size) != record["size"]:
            raise ValueError("Label/manifest mismatch")
        valid_points(points)
        seen_ids.add(record["id"])
        seen_pixels.add(pixel)
        hashes[split].append(dhash)
    actual = {p.relative_to(root).as_posix() for folder in ("images", "labels") for p in (root / folder).rglob("*") if p.is_file()}
    if actual != expected or not all(hashes.values()):
        raise ValueError("Unexpected/missing files or empty split")
    if any((a ^ b).bit_count() <= manifest["dhash_distance"] for a in hashes["train"] for b in hashes["val"]):
        raise ValueError("Near-duplicate leakage")
    print(f"Verified {len(seen_ids)} images; no exact/pixel, group, or configured dHash overlap")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("scan")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--propose-contours", action="store_true")
    p = commands.add_parser("finalize")
    p.add_argument("--workspace", required=True)
    p.add_argument("--reviews", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--val-fraction", type=float, default=.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dhash-distance", type=int, default=4)
    p = commands.add_parser("verify")
    p.add_argument("--root", required=True)
    p = commands.add_parser("plan")
    p.add_argument("--workspace", required=True)
    args = parser.parse_args()
    if args.command == "scan":
        scan(args.source, args.out, args.propose_contours)
    elif args.command == "finalize":
        finalize(args.workspace, args.reviews, args.out, args.val_fraction, args.seed, args.dhash_distance)
    elif args.command == "plan":
        plan = draft_plan(args.workspace)
        print(json.dumps({key: value for key, value in plan.items() if key != "assignments"}, indent=2))
    else:
        verify(args.root)


if __name__ == "__main__":
    main()
