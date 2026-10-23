# Training workflow decisions

## ANPR review tooling interactions — 2026-10-21 (Approved)

2026-10-22 owner approved Drive checkpoint persistence/resume and training visibility in the Colab notebook. Keep the dataset local for training, cache its ZIP on the user's Drive, verify run contract and checkpoint hashes, and commit two rotating checkpoint generations after completed epochs while preserving optimizer state before Ultralytics strips final local weights. Resume from the latest verified epoch; an interrupted in-progress epoch may repeat. Add native logs, a periodic status dashboard and TensorBoard, without claiming that a stale heartbeat proves runtime liveness. Do not upload or train real data as part of local tooling verification; synthetic CPU interruption/resume tests are allowed.

2026-10-22 owner requested an unreviewed-image filter. The toolbar offers all, unreviewed and reviewed views; navigation, keyboard shortcuts and approval progression respect the selected filter and always omit excluded images. Save before changing filters, stop on save conflict and display a recoverable empty state when no images match. Final approval must retain a valid server cursor even when the filtered list becomes empty. Filtering does not modify labels or group assignments.

2026-10-22 owner approved keyboard shortcuts: Enter invokes Duyệt, ArrowLeft invokes previous image and ArrowRight invokes next image. Shortcuts reuse existing save/approval actions; ignore editable controls, IME composition, modified keys, key repeats and actions while saving/drawing to prevent accidental approvals or skipped images.

Owner approved three annotation modes: individual corner clicks, drag an axis-aligned rectangle, and drag a square. Both drag modes emit four normalized corners in TL/TR/BR/BL order; square dimensions are equal in source-image pixels, including at display scaling and image bounds. Preview/cancel do not change stored annotations; a completed nondegenerate drag clears approval and uses the existing save workflow. These modes assist annotation and do not automatically approve geometry or substitute for perspective corners on tilted plates.

Owner requested a compact layout: reviewer and group inputs share the header action row with buttons, all controls remain 28px high, spacing is reduced and the toolbar wraps on narrow screens. Annotation and persistence behavior are unchanged.

Follow-up owner request: the action row uses Duyệt, Lưu and Loại. Duyệt advances only after successful save and persists the next-image cursor; Lưu stays on the current image. Remove separate exclusion/group-confirmation checkboxes, the import label and autosave help line; hide the unused import control on LAN. Keep autosave and explicit save/error status. Duplicate approval clicks during a save must not advance or approve another image.

Owner requested that approving an image also confirms its current leakage group, without a separate checkbox step. Four points, a nonempty group and reviewer remain required; this action records the operator's confirmation and does not infer group correctness. On the LAN review tool, excluding an image immediately saves the exclusion, moves the image to recoverable `excluded-images/` inside the external quarantine workspace, removes HTTP image access and skips it during navigation. Original source archives remain intact. LAN credentials were changed to the owner-requested values outside Git; secret values are not recorded here. These are review-tool changes only, with no Gate runtime, product database, training or deployment behavior change.

### 2026-10-22 — Approved offline ANPR mixed-source training package v2

Owner requested a new four-corner training package containing both cars and
motorcycles, preserving group leakage controls and Drive checkpoint support.
Keep whole reviewed source/alias groups and near-duplicate components: car and
green train, bv1/cam validation. Do not invent vehicle/session identities to
force an 80/20 split. Validation is motorcycle-only and is not an independent
car acceptance set. The old car validation becomes training data and cannot be
reused as an independent v2 test. Preserve all reviewed corners, exclusions,
the original package and the running v1 job. Generate a separate notebook with
version-specific Drive/cache, pinned package/manifest and checkpoint helpers
embedded at the point of use. No Gate runtime, ACS implementation, production
database, deployment or new runtime capability is included. See
docs/anpr-pose-mixed-v2.md; generated data stays outside Git.

### 2026-10-22 — Human-reviewed car plate groups before mixed validation

Owner approved local OCR suggestions and a LAN review page for the 851 car-source
images. OCR does not approve identities. Require per-image plate/group approval
and reviewer before creating a split plan. Join equal reviewed plates, optional
manual groups, byte/pixel duplicates and configured dHash edges; preserve whole
motorcycle source groups. Both train and validation must contain both vehicle
types. Keep review state and OCR text outside Git with source-manifest binding,
revision conflicts and persisted history. Package v3 only after human review;
train from pretrained because v2 has already trained on the proposed car holdout.
See docs/anpr-car-group-review.md. No runtime capability or database change.
