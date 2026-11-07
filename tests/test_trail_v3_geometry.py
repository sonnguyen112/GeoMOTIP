import torch

from models.motip.geometry import (
    GEOMETRY_FEATURE_DIM,
    GeometryCompatibilityBranch,
    build_g4_features,
    build_pairwise_g4_features,
)
from models.motip.id_decoder import (
    IDDecoder,
    _candidate_geometry_logit_diagnostics,
    _geometry_logit_scale_diagnostics,
)
from models.runtime_tracker import RuntimeTracker
from submit_and_evaluate import save_branch_logit_cache


def _seq(batch=1, groups=1, times=5, slots=3, dim=16, vocab=8):
    time = torch.arange(times).view(1, 1, times, 1).expand(batch, groups, times, slots)
    labels = torch.arange(slots).view(1, 1, 1, slots).expand(batch, groups, times, slots).clone()
    boxes = torch.zeros(batch, groups, times, slots, 4)
    boxes[..., 0] = time.float() / max(times, 1)
    boxes[..., 1] = torch.arange(slots).view(1, 1, 1, slots) / max(slots, 1)
    boxes[..., 2:] = 0.1
    masks = torch.zeros(batch, groups, times, slots, dtype=torch.bool)
    unknown = {
        "unknown_features": torch.randn(batch, groups, times, slots, dim),
        "unknown_boxes": boxes.clone(),
        "unknown_times": time.clone(),
        "unknown_masks": masks.clone(),
        "unknown_id_labels": labels.clone(),
    }
    history = {
        "trajectory_features": torch.randn(batch, groups, times, slots, dim),
        "trajectory_boxes": boxes,
        "trajectory_id_labels": labels,
        "trajectory_times": time.clone(),
        "trajectory_masks": masks,
    }
    return {**history, **unknown}


def test_g4_dimension_and_exact_lag_masks():
    current = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    history = current[:, None].expand(1, 4, 4).clone()
    valid = torch.ones(1, 4, dtype=torch.bool)
    assert build_g4_features(current, history, valid).shape[-1] == GEOMETRY_FEATURE_DIM

    current_boxes = torch.tensor([[[[[0.5, 0.5, 0.2, 0.4]]]]])
    current_times = torch.tensor([[[[4]]]])
    history_boxes = torch.zeros(1, 1, 5, 1, 4)
    history_times = torch.arange(5).view(1, 1, 5, 1)
    history_masks = torch.zeros(1, 1, 5, 1, dtype=torch.bool)
    _, validities = build_pairwise_g4_features(
        current_boxes, current_times, history_boxes, history_times, history_masks
    )
    assert validities.shape == (1, 1, 1, 1, 1, 4)
    assert validities.flatten().tolist() == [True, True, True, False]


def test_missing_lag_is_not_nearest_frame_substitution_or_future_leakage():
    current_boxes = torch.tensor([[[[[0.5, 0.5, 0.2, 0.4]]]]])
    current_times = torch.tensor([[[[2]]]])
    history_boxes = torch.rand(1, 1, 4, 1, 4)
    history_times = torch.tensor([[[[0], [1], [2], [3]]]])
    history_masks = torch.zeros(1, 1, 4, 1, dtype=torch.bool)
    features, validities = build_pairwise_g4_features(
        current_boxes, current_times, history_boxes, history_times, history_masks
    )
    # t-1 exists, t-2 exists, t-4 and t-8 do not; t+1 is not used.
    assert validities.flatten().tolist() == [True, True, False, False]
    assert torch.count_nonzero(features[..., 4:20]) > 0
    assert torch.count_nonzero(features[..., 4 + 4 : 4 + 8]) > 0


def test_geometry_prefers_absolute_frame_timestamps_over_local_positions():
    current_boxes = torch.tensor([[[[[0.5, 0.5, 0.2, 0.4]]]]])
    # The retained history is sampled at actual frames 10, 14, and 18.  The
    # local MOTIP positions (0, 1, 2) do not represent those frame gaps.
    history_boxes = torch.rand(1, 1, 3, 1, 4)
    history_masks = torch.zeros(1, 1, 3, 1, dtype=torch.bool)
    local_times = torch.tensor([[[[0], [1], [2]]]])
    absolute_times = torch.tensor([[[[10], [14], [18]]]])
    current_time = torch.tensor([[[[18]]]])
    _, local_valid = build_pairwise_g4_features(
        current_boxes, current_time, history_boxes, local_times, history_masks
    )
    _, absolute_valid = build_pairwise_g4_features(
        current_boxes, current_time, history_boxes, absolute_times, history_masks
    )
    assert local_valid.flatten().tolist() == [False, False, False, False]
    assert absolute_valid.flatten().tolist() == [False, False, True, True]


def test_geometry_branch_shapes_newborn_zero_and_auxiliary_gradient():
    seq = _seq()
    # Remove all history for the first query; no geometry score can be assigned.
    seq["trajectory_masks"][:, :, 0] = True
    branch = GeometryCompatibilityBranch(num_id_vocabulary=8)
    logits, pair_scores, pair_valid = branch(seq)
    assert logits.shape == (1, 1, 5, 3, 9)
    assert pair_scores.shape == (1, 1, 5, 3, 3)
    assert pair_valid.shape == pair_scores.shape
    assert torch.equal(logits[:, :, 0, :, :], torch.zeros_like(logits[:, :, 0, :, :]))
    branch.last_geometry_loss.backward()
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        for p in branch.parameters()
    )


def test_geometry_scale_diagnostic_accounts_for_fusion_temperatures():
    base = torch.tensor([[[[3.0, 4.0]]]])
    geometry = torch.tensor([[[[6.0, 8.0]]]])
    stats = _geometry_logit_scale_diagnostics(
        base, geometry, torch.tensor(0.5), base_temperature=2.0, geometry_temperature=0.5
    )
    assert abs(stats["contribution_ratio"] - 1.0) < 1e-5
    assert abs(stats["calibrated_base_logit_norm"] - 2.5) < 1e-5
    assert abs(stats["calibrated_geometry_logit_norm"] - 10.0) < 1e-5
    assert abs(stats["calibrated_contribution_ratio"] - 4.0) < 1e-5


def test_candidate_geometry_diagnostics_mask_inactive_vocabulary_and_include_new():
    # Five logits: four active-ID vocabulary slots plus NEW at index four.
    # Only IDs 0 and 1 are active; extreme ID-2/ID-3 values must not distort
    # candidate calibration statistics. NEW is retained in every query set.
    base = torch.tensor([[[[[5.0, 4.0, 900.0, -800.0, 1.0]]]]])
    geometry = torch.tensor([[[[[0.0, 2.0, 0.0, 0.0, 0.0]]]]])
    candidate_ids = torch.tensor([[[[[0, 1, 2]]]]])
    candidate_active = torch.tensor([[[[[True, True, False]]]]])
    unknown_masks = torch.zeros(1, 1, 1, 1, dtype=torch.bool)

    stats = _candidate_geometry_logit_diagnostics(
        base,
        geometry,
        candidate_ids,
        candidate_active,
        unknown_masks,
        torch.tensor(1.0),
        base_temperature=2.0,
        geometry_temperature=0.5,
    )
    assert stats["candidate_count_mean"] == 3.0  # IDs 0, 1, and NEW.
    assert stats["geometry_candidate_centered_ratio"] > 0
    assert stats["geometry_calibrated_candidate_centered_ratio"] > stats[
        "geometry_candidate_centered_ratio"
    ]

    changed_inactive = base.clone()
    changed_inactive[..., 2] = -1e6
    changed_inactive[..., 3] = 1e6
    repeated = _candidate_geometry_logit_diagnostics(
        changed_inactive,
        geometry,
        candidate_ids,
        candidate_active,
        unknown_masks,
        torch.tensor(1.0),
        base_temperature=2.0,
        geometry_temperature=0.5,
    )
    assert repeated == stats


def test_candidate_geometry_diagnostics_broadcast_over_auxiliary_layers():
    base = torch.tensor([[[[[[4.0, 2.0, 1.0]]]]], [[[[[3.0, 1.0, 0.0]]]]]])
    geometry = torch.tensor([[[[[[1.0, 0.0, 0.0]]]]], [[[[[0.0, 1.0, 0.0]]]]]])
    candidate_ids = torch.tensor([[[[[0, 1]]]]])
    candidate_active = torch.tensor([[[[[True, True]]]]])
    unknown_masks = torch.zeros(1, 1, 1, 1, dtype=torch.bool)
    stats = _candidate_geometry_logit_diagnostics(
        base,
        geometry,
        candidate_ids,
        candidate_active,
        unknown_masks,
        torch.tensor(1.0),
        base_temperature=1.0,
        geometry_temperature=1.0,
    )
    assert stats["candidate_count_mean"] == 3.0
    assert torch.isfinite(torch.tensor(list(stats.values()))).all()


def test_iddecoder_candidate_diagnostics_are_optional_and_do_not_change_logits():
    torch.manual_seed(17)
    seq = _seq(times=4, slots=3, dim=16, vocab=8)
    common = dict(
        feature_dim=16,
        id_dim=16,
        ffn_dim_ratio=2,
        num_layers=2,
        head_dim=16,
        num_id_vocabulary=8,
        rel_pe_length=30,
        use_aux_loss=False,
        use_shared_aux_head=True,
        use_geometry=True,
        geometry_fixed_lambda=1.0,
    )
    plain = IDDecoder(**common, geometry_candidate_diagnostics=False).eval()
    diagnostic = IDDecoder(**common, geometry_candidate_diagnostics=True).eval()
    diagnostic.load_state_dict(plain.state_dict())
    plain_logits = plain(seq, False)[0]
    diagnostic_logits = diagnostic(seq, False)[0]
    torch.testing.assert_close(plain_logits, diagnostic_logits, rtol=0, atol=0)
    assert plain.geometry_candidate_diagnostics == {}
    assert plain.geometry_branch.last_candidate_ids is None
    assert plain.geometry_branch.last_candidate_active is None
    assert plain.geometry_branch.last_exact_lag_valid is None
    assert diagnostic.geometry_branch.last_candidate_ids is not None
    assert diagnostic.geometry_branch.last_exact_lag_valid is not None
    assert diagnostic.geometry_branch.last_exact_lag_valid.shape[-1] == 4
    assert "geometry_calibrated_candidate_centered_ratio" in diagnostic.geometry_candidate_diagnostics
    assert diagnostic.geometry_candidate_diagnostics["candidate_count_mean"] == 4.0


def test_v3_zero_lambda_reproduces_baseline_logits_exactly():
    torch.manual_seed(12)
    seq = _seq(dim=16)
    baseline = IDDecoder(16, 16, 2, 2, 16, 8, 30, False, True)
    geometry = IDDecoder(
        16,
        16,
        2,
        2,
        16,
        8,
        30,
        False,
        True,
        use_geometry=True,
        geometry_fixed_lambda=0.0,
    )
    missing, unexpected = geometry.load_state_dict(baseline.state_dict(), strict=False)
    assert all(name.startswith("geometry_branch.") for name in missing)
    assert not unexpected
    baseline.eval()
    geometry.eval()
    expected = baseline(seq, False)[0]
    actual = geometry(seq, False)[0]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_v3_freezes_motip_and_geometry_receives_gradient():
    torch.manual_seed(13)
    seq = _seq(dim=16)
    decoder = IDDecoder(
        16,
        16,
        2,
        2,
        16,
        8,
        30,
        False,
        True,
        use_geometry=True,
        geometry_fixed_lambda=0.01,
        geometry_freeze_baseline=True,
    )
    decoder.train()
    logits, _, _ = decoder(seq, False)
    (logits.square().mean() + decoder.geometry_aux_loss).backward()
    geometry_ids = {id(p) for p in decoder.geometry_branch.parameters()}
    for name, parameter in decoder.named_parameters():
        if id(parameter) in geometry_ids:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None


def test_candidate_local_ids_and_newborn_class_are_not_geometry_scored():
    seq = _seq(times=3, slots=2, dim=16)
    # The second slot has no pre-current observation at t=1; it must not get a
    # geometry score even though its local label is valid in the label table.
    seq["trajectory_masks"][:, :, 0, 1] = True
    branch = GeometryCompatibilityBranch(num_id_vocabulary=8)
    logits, _, valid = branch(seq)
    assert not valid[0, 0, 1, :, 1].any()
    assert torch.equal(logits[..., 8], torch.zeros_like(logits[..., 8]))


def test_candidate_metadata_stays_aligned_when_runtime_drops_detections(tmp_path):
    tracker = RuntimeTracker.__new__(RuntimeTracker)
    tracker.last_base_logits = torch.arange(15).reshape(1, 1, 1, 3, 5)
    tracker.last_geometry_logits = tracker.last_base_logits + 100
    tracker.last_fused_logits = tracker.last_base_logits + 200
    tracker.last_geometry_candidate_ids = torch.tensor([[[[[0, 1], [1, 2], [2, 3]]]]])
    tracker.last_geometry_candidate_active = torch.ones_like(
        tracker.last_geometry_candidate_ids, dtype=torch.bool
    )
    tracker.last_geometry_candidate_valid = tracker.last_geometry_candidate_active.clone()
    # This tensor has an extra trailing lag axis: [..., detection, candidate, lag].
    tracker.last_geometry_candidate_lag_valid = (
        torch.arange(24).reshape(1, 1, 1, 3, 2, 4).remainder(2).bool()
    )
    expected_lag_valid = tracker.last_geometry_candidate_lag_valid.index_select(
        -3, torch.tensor([0, 2])
    )
    tracker._filter_last_decoder_logits(torch.tensor([True, False, True]))

    expected_ids = torch.tensor([[[[[0, 1], [2, 3]]]]])
    assert torch.equal(tracker.last_geometry_candidate_ids, expected_ids)
    assert torch.equal(tracker.last_geometry_candidate_lag_valid, expected_lag_valid)
    assert torch.equal(tracker.last_base_logits, torch.arange(15).reshape(1, 1, 3, 5)[:, :, [0, 2]].unsqueeze(1))

    frame = {
        "id": torch.tensor([4, 5]),
        "slot_id": torch.tensor([0, 2]),
        "bbox": torch.zeros(2, 4),
        "base_logits": tracker.last_base_logits[0, 0, 0],
        "geometry_logits": tracker.last_geometry_logits[0, 0, 0],
        "fused_logits": tracker.last_fused_logits[0, 0, 0],
        "geometry_candidate_ids": tracker.last_geometry_candidate_ids[0, 0, 0],
        "geometry_candidate_active": tracker.last_geometry_candidate_active[0, 0, 0],
        "geometry_candidate_valid": tracker.last_geometry_candidate_valid[0, 0, 0],
    }
    path = save_branch_logit_cache([frame], tmp_path, "seq")
    cached = torch.load(path, map_location="cpu", weights_only=False)["frames"][0]
    assert torch.equal(cached["geometry_candidate_ids"], expected_ids[0, 0, 0])
    assert torch.equal(cached["base_logits"], frame["base_logits"])


def test_candidate_cache_prefers_prefilter_detector_cohort(tmp_path):
    candidate_eval = {
        "base_logits": torch.tensor([[0.0, 1.0, 2.0]]),
        "geometry_logits": torch.tensor([[0.0, 2.0, 0.0]]),
        "fused_logits": torch.tensor([[0.0, 5.0, 2.0]]),
        "candidate_ids": torch.tensor([[0, 1]]),
        "candidate_active": torch.tensor([[True, True]]),
        "candidate_valid": torch.tensor([[True, False]]),
        "candidate_lag_valid": torch.tensor([[[True, False, False, False], [False] * 4]]),
        "boxes_xywh": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        "detection_scores": torch.tensor([0.75]),
        "pre_filter_id_prediction": torch.tensor([2]),
    }
    frame = {
        "id": torch.tensor([9]),
        "slot_id": torch.tensor([2]),
        "bbox": torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        "base_logits": torch.tensor([[99.0, 99.0, 99.0]]),
        "geometry_logits": torch.tensor([[99.0, 99.0, 99.0]]),
        "fused_logits": torch.tensor([[99.0, 99.0, 99.0]]),
        "candidate_eval": candidate_eval,
    }
    path = save_branch_logit_cache([frame], tmp_path, "seq")
    cached = torch.load(path, map_location="cpu", weights_only=False)["frames"][0]
    assert "base_logits" not in cached  # filtered branch rows are not duplicated
    assert torch.equal(cached["candidate_eval"]["base_logits"], candidate_eval["base_logits"])
    assert torch.equal(
        cached["candidate_eval"]["candidate_lag_valid"],
        candidate_eval["candidate_lag_valid"],
    )
