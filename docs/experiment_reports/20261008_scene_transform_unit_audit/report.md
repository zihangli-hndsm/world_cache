# Confirmed 3RScan scene-transform unit correction

The released metadata uses metre translations. The local parser had multiplied
translations by 0.001 following a conflicting FAQ statement. Official data.cc
reads matrices unchanged; rio.cc applies them directly to metre mesh vertices;
sequence.cc multiplies metadata translations by 1000 only when requesting poses
in millimetres. Pinned official sources and hashes are retained here.

An evaluation-only static-surface audit checks all 110 validation relations,
excluding both sides of every annotated rigid-object motion. No registration is
fit and no scale is selected per relation. Raw metadata improves 5 cm support
in 110/110 relations, and median distance in 109/110. The median relation-level
nearest-surface distance is 0.0391 m raw versus 0.3649 m under the former scaling.
The one median-distance exception does not justify a different per-scene unit.

Exact cached ViT-G consensus reproduction validates all selection, inlier,
residual, and old pose-error values before rescoring. Corrected strict success:
14/110 validation relations (previously 2/110), and 4/16 source relations
(previously 1/16). The target remains successful, with translation error 0.0374 m
(previously 0.1718 m). Native-coordinate predictions and semantic/instance
transfer counts are unchanged. Reports:
- outputs/reports/20261008_vitg_validation_corrected_pose_units/
- outputs/reports/20261008_vitg_source_corrected_pose_units/

The full-validation object-vs-scene valid-K20 oracle gap at 10 cm falls from
+18.6 pp to +4.7 pp [1.6,8.6]. At 5/20 cm it is +3.3 [1.2,5.7] and +4.0
[1.0,8.1]. Object-valid subsets and conditional identity-correctness deltas
are unchanged, since the per-object oracle was already in metres.
Corrected oracle source/validation outputs have corrected_units in their paths.

The paper remains under correction. paper/SCENE_UNIT_CORRECTION_PENDING.md
blocks verification and submission bundling until all dependent claims,
figures, baseline scores and generated tables have been audited and refreshed.
Existing PDFs/ZIP and prior verification-pass files describe an obsolete snapshot.

BUFFER-X inference continues because GT units do not affect its native input
or prediction. The final aggregate rescoring reads persisted prediction matrices
and applies one corrected unit convention to every scheduled relation.
