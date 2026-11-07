# Copyright (c) Ruopeng Gao. All Rights Reserved.
# About: Submit or evaluate the model.

import os
import time
import torch
import subprocess
import sys
import json
from accelerate import Accelerator
from accelerate.state import PartialState
from torch.utils.data import DataLoader

from runtime_option import runtime_option
from utils.misc import yaml_to_dict
from configs.util import load_super_config, update_config
from log.logger import Logger
from data.joint_dataset import dataset_classes
from data.seq_dataset import SeqDataset
from data.sequence_manifest import filter_inference_dataset, write_trackeval_seqmap
from models.runtime_tracker import RuntimeTracker
from log.log import Metrics
from models.motip import build as build_motip
from models.misc import load_checkpoint
from utils.history_write_risk import LogisticHistoryCommitPolicy


def save_branch_logit_cache(sequence_results, output_dir, sequence_name, metadata=None):
    """Persist per-frame association branch outputs for an opt-in diagnostic replay.

    The normal MOTChallenge writer intentionally ignores these internal fields.
    This cache contains predictions only (no GT labels), so a later analysis can
    inspect branch scales/agreement without changing the standard submission.
    When available, ``candidate_eval`` is the detector-thresholded cohort
    captured before ID-dependent newborn filtering, enabling paired branch
    comparisons on exactly the same detector proposals.
    """
    import os

    branch_keys = ("base_logits", "geometry_logits", "fused_logits")
    candidate_keys = (
        "geometry_candidate_ids", "geometry_candidate_active",
        "geometry_candidate_valid", "geometry_candidate_lag_valid",
    )
    diagnostic_keys = ("history_commit_mask",)
    frames = []
    found_branch_logits = False
    for frame_index, result in enumerate(sequence_results):
        frame = {"frame_id": frame_index + 1}
        candidate_eval = result.get("candidate_eval")
        for key in (
            "id", "slot_id", "id_confidence", "score", "category", "bbox",
            *branch_keys, *candidate_keys, *diagnostic_keys,
        ):
            # Prefer the pre-filter, detector-thresholded cohort when present;
            # the normal tracker fields below it may have dropped low-score
            # detections after the ID branch predicted NEW.
            if candidate_eval is not None and key in (*branch_keys, *candidate_keys):
                continue
            value = result.get(key)
            if torch.is_tensor(value):
                value = value.detach().cpu()
            frame[key] = value
        if candidate_eval is not None:
            frame["candidate_eval"] = {
                key: value.detach().cpu() if torch.is_tensor(value) else value
                for key, value in candidate_eval.items()
            }
        frame["geometry_diagnostics"] = result.get("geometry_diagnostics")
        found_branch_logits |= (
            any(torch.is_tensor(frame.get(key)) for key in branch_keys)
            or (
                isinstance(frame.get("candidate_eval"), dict)
                and torch.is_tensor(frame["candidate_eval"].get("base_logits"))
            )
        )
        frames.append(frame)
    if not found_branch_logits:
        return None

    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "sequence": sequence_name,
        "metadata": dict(metadata or {}),
        "frames": frames,
    }
    target_path = os.path.join(output_dir, f"{sequence_name}.pt")
    temporary_path = target_path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, target_path)
    return target_path


def submit_and_evaluate(config: dict):
    # Init Accelerator at beginning:
    # Match the training-time evaluation autocast policy in standalone runs.
    accelerator = Accelerator(mixed_precision=config.get("MIXED_PRECISION", None))
    state = PartialState()

    mode = config["INFERENCE_MODE"]
    assert mode in ["submit", "evaluate"], f"Mode {mode} is not supported."
    # Generate the output dir:
    assert "OUTPUTS_DIR" in config and config["OUTPUTS_DIR"] is not None, "OUTPUTS_DIR is not set."
    outputs_dir = config["OUTPUTS_DIR"]
    inference_group = config["INFERENCE_GROUP"]
    inference_dataset = config["INFERENCE_DATASET"]
    inference_split = config["INFERENCE_SPLIT"]
    inference_model = config["INFERENCE_MODEL"]
    _inference_model_name = os.path.split(inference_model)[-1][:-4]
    outputs_dir = os.path.join(
        outputs_dir, mode, inference_group, inference_dataset, inference_split, _inference_model_name
    )
    _is_outputs_dir_exist = os.path.exists(outputs_dir)
    accelerator.wait_for_everyone()
    os.makedirs(outputs_dir, exist_ok=True)

    # Init Logger, do not use wandb:
    logger = Logger(
        logdir=str(outputs_dir),
        use_wandb=False,
        config=config,
        # exp_owner=config["EXP_OWNER"],
        # exp_project=config["EXP_PROJECT"],
        # exp_group=config["EXP_GROUP"],
        # exp_name=config["EXP_NAME"],
    )
    # Log runtime config:
    logger.config(config=config)
    # Log other infos:
    logger.info(
        f"{mode.capitalize()} model: {inference_model}, inference dataset: {inference_dataset}, "
        f"inference split: {inference_split}, inference group: {inference_group}."
    )
    if _is_outputs_dir_exist:
        logger.warning(f"Outputs dir '{outputs_dir}' already exists, may overwrite the existing files.")
        time.sleep(5)   # wait for 5 seconds, give the user a chance to cancel.
    else:
        logger.info(f"Outputs dir '{outputs_dir}' created.")

    model, _ = build_motip(config=config)

    use_previous_checkpoint = config.get("USE_PREVIOUS_CHECKPOINT", False)
    if not use_previous_checkpoint:
        load_checkpoint(model, path=config["INFERENCE_MODEL"])
    else:
        from models.misc import load_previous_checkpoint
        load_previous_checkpoint(model, path=config["INFERENCE_MODEL"])

    model = accelerator.prepare(model)

    metrics = submit_and_evaluate_one_model(
        is_evaluate=config["INFERENCE_MODE"] == "evaluate",
        accelerator=accelerator,
        state=state,
        logger=logger,
        model=model,
        data_root=config["DATA_ROOT"],
        dataset=config["INFERENCE_DATASET"],
        data_split=config["INFERENCE_SPLIT"],
        outputs_dir=outputs_dir,
        image_max_longer=config["INFERENCE_MAX_LONGER"],    # the max shorter side of the image is set to 800 by default
        image_max_shorter=config.get("INFERENCE_MAX_SHORTER", 800),
        size_divisibility=config.get("SIZE_DIVISIBILITY", 0),
        use_sigmoid=config.get("USE_FOCAL_LOSS", False),
        assignment_protocol=config.get("ASSIGNMENT_PROTOCOL", "hungarian"),
        miss_tolerance=config["MISS_TOLERANCE"],
        det_thresh=config["DET_THRESH"],
        newborn_thresh=config["NEWBORN_THRESH"],
        id_thresh=config["ID_THRESH"],
        area_thresh=config.get("AREA_THRESH", 0),
        inference_only_detr=config["INFERENCE_ONLY_DETR"] if config["INFERENCE_ONLY_DETR"] is not None
        else config["ONLY_DETR"],
        dtype=config.get("INFERENCE_DTYPE", "FP32"),
        sequence_manifest=config.get("INFERENCE_SEQUENCE_MANIFEST"),
        config=config,
    )

    if metrics is not None:
        metrics.sync()
        logger.metrics(
            log=f"Finish evaluation for model '{inference_model}', dataset '{inference_dataset}', "
                f"split '{inference_split}', group '{inference_group}': ",
            metrics=metrics,
            fmt="{global_average:.4f}",
        )
    return


def submit_and_evaluate_one_model(
        is_evaluate: bool,
        accelerator: Accelerator,
        state: PartialState,
        logger: Logger,
        model,
        data_root: str,
        dataset: str,
        data_split: str,
        # Outputs:
        outputs_dir: str,
        # Parameters with defaults:
        image_max_shorter: int = 800,
        image_max_longer: int = 1536,
        size_divisibility: int = 0,
        use_sigmoid: bool = False,
        assignment_protocol: str = "hungarian",
        miss_tolerance: int = 30,
        det_thresh: float = 0.5,
        newborn_thresh: float = 0.5,
        id_thresh: float = 0.1,
        area_thresh: int = 0,
        inference_only_detr: bool = False,
        dtype: str = "FP32",
        sequence_manifest: str | None = None,
        config: dict | None = None,
):
    config = config or {}
    # Build the datasets:
    inference_dataset = dataset_classes[dataset](
        data_root=data_root,
        split=data_split,
        load_annotation=False,
    )
    selected_sequence_names = filter_inference_dataset(inference_dataset, sequence_manifest)
    if selected_sequence_names is None:
        selected_sequence_names = sorted(inference_dataset.sequence_infos.keys())
    # A subset must use a matching TrackEval seqmap; the full map would
    # request tracker files for sequences deliberately not evaluated.
    subset_seqmap_path = None
    if sequence_manifest is not None:
        subset_seqmap_path = write_trackeval_seqmap(
            os.path.join(outputs_dir, f"{data_split}_seqmap.txt"), selected_sequence_names
        )
    # Set the dtype during inference:
    match dtype:
        case "FP32": dtype=torch.float32
        case "FP16": dtype=torch.float16
        case _: raise ValueError(f"Unknown dtype '{dtype}'.")
    # Filter out the sequences that will not be processed in this GPU (if we have multiple GPUs):
    _inference_sequence_names = list(inference_dataset.sequence_infos.keys())
    _inference_sequence_names.sort()
    # If we have multiple GPUs, we need to filter out the sequences that will not be processed in this GPU:
    # However, there is a special case that the number of GPUs is larger than the number of sequences:
    if len(_inference_sequence_names) <= state.process_index:
        logger.info(
            log=f"Number of sequences is smaller than the number of processes, "
                f"a fake sequence will be processed on process {state.process_index}.",
            only_main=False,
        )
        inference_dataset.sequence_infos = {
            _inference_sequence_names[0]: inference_dataset.sequence_infos[_inference_sequence_names[0]]
        }
        inference_dataset.image_paths = {
            _inference_sequence_names[0]: inference_dataset.image_paths[_inference_sequence_names[0]]
        }
        is_fake = True
    else:
        for _ in range(len(_inference_sequence_names)):
            if _ % state.num_processes != state.process_index:
                inference_dataset.sequence_infos.pop(_inference_sequence_names[_])
                inference_dataset.image_paths.pop(_inference_sequence_names[_])
        is_fake = False

    # Process each sequence:
    performance = []
    history_commit_policy = None
    gate_model_path = config.get("HISTORY_COMMIT_GATE_MODEL")
    if gate_model_path:
        gate_threshold = config.get("HISTORY_COMMIT_GATE_THRESHOLD")
        if gate_threshold is None:
            raise ValueError("HISTORY_COMMIT_GATE_THRESHOLD is required with HISTORY_COMMIT_GATE_MODEL")
        with open(gate_model_path, "r") as gate_stream:
            gate_payload = json.load(gate_stream)
        history_commit_policy = LogisticHistoryCommitPolicy(
            gate_payload, threshold=float(gate_threshold),
        )
    for sequence_name in inference_dataset.sequence_infos.keys():
        # break
        sequence_dataset = SeqDataset(
            seq_info=inference_dataset.sequence_infos[sequence_name],
            image_paths=inference_dataset.image_paths[sequence_name],
            max_shorter=image_max_shorter,
            max_longer=image_max_longer,
            size_divisibility=size_divisibility,
            dtype=dtype,
        )
        sequence_loader = DataLoader(
            dataset=sequence_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=lambda x: x[0],
        )
        # sequence_loader = accelerator.prepare(sequence_loader)
        sequence_wh = sequence_dataset.seq_hw()
        runtime_tracker = RuntimeTracker(
            model=model,
            sequence_hw=sequence_wh,
            use_sigmoid=use_sigmoid,
            assignment_protocol=assignment_protocol,
            miss_tolerance=miss_tolerance,
            det_thresh=det_thresh,
            newborn_thresh=newborn_thresh,
            id_thresh=id_thresh,
            area_thresh=area_thresh,
            only_detr=inference_only_detr,
            dtype=dtype,
            save_candidate_appearance_diagnostics=bool(
                config.get("SAVE_CANDIDATE_APPEARANCE_DIAGNOSTICS", False)
            ),
            history_commit_policy=history_commit_policy,
            tracklet_confirm_n_init=config.get("TRACKLET_CONFIRM_N_INIT", 1),
        )
        if is_fake:
            logger.info(
                f"Fake submitting sequence {sequence_name} with {len(sequence_loader)} frames.",
                only_main=False
            )
        else:
            logger.info(f"Submitting sequence {sequence_name} with {len(sequence_loader)} frames.", only_main=False)
        sequence_results, sequence_fps = get_results_of_one_sequence(
            runtime_tracker=runtime_tracker,
            sequence_loader=sequence_loader,
            logger=logger,
        )
        save_branch_logits = config.get("SAVE_BRANCH_LOGITS", False) or (
            os.environ.get("MOTIP_SAVE_BRANCH_LOGITS", "0").lower()
            in {"1", "true", "yes"}
        )
        if save_branch_logits:
            branch_cache_dir = os.environ.get(
                "MOTIP_BRANCH_LOGITS_DIR",
                config.get("BRANCH_LOGITS_DIR") or os.path.join(outputs_dir, "branch_logits"),
            )
            cache_metadata = {
                "checkpoint": config.get("INFERENCE_MODEL"),
                "dataset": dataset,
                "split": data_split,
                "base_temperature": config.get("GEOMETRY_BASE_TEMPERATURE"),
                "geometry_temperature": config.get("GEOMETRY_TEMPERATURE"),
                "fixed_lambda": config.get("GEOMETRY_FIXED_LAMBDA"),
                "geometry_lags": config.get("GEOMETRY_LAGS"),
                "num_id_vocabulary": config.get("NUM_ID_VOCABULARY"),
                "new_class_index": config.get("NUM_ID_VOCABULARY"),
                "detector_threshold": config.get("DET_THRESH"),
                "newborn_threshold": config.get("NEWBORN_THRESH"),
                "id_threshold": config.get("ID_THRESH"),
                "assignment_protocol": config.get("ASSIGNMENT_PROTOCOL"),
                "use_sigmoid_id_scores": bool(config.get("USE_FOCAL_LOSS", False)),
                "candidate_eval_cohort": "all detector-thresholded queries before ID-dependent newborn filtering",
                "candidate_appearance_diagnostics": bool(
                    config.get("SAVE_CANDIDATE_APPEARANCE_DIAGNOSTICS", False)
                ),
            }
            saved_cache = save_branch_logit_cache(
                sequence_results, branch_cache_dir, sequence_name, cache_metadata
            )
            if saved_cache is not None:
                logger.info(f"Saved branch-logit diagnostic cache: {saved_cache}", only_main=False)
        performance.append({"sequence": sequence_name, "frames": len(sequence_results),
                            "timed_frames": len(sequence_results) - 10, "fps": sequence_fps})
        # Write the results to the submit file:
        if dataset in ["DanceTrack", "SportsMOT", "MOT17", "PersonPath22_Inference", "BFT"]:
            sequence_tracker_results = []
            for t in range(len(sequence_results)):
                for obj_id, score, category, bbox in zip(
                        sequence_results[t]["id"],
                        sequence_results[t]["score"],
                        sequence_results[t]["category"],
                        sequence_results[t]["bbox"],    # [x, y, w, h]
                ):
                    sequence_tracker_results.append(
                        f"{t + 1},{obj_id.item()},"
                        f"{bbox[0].item()},{bbox[1].item()},{bbox[2].item()},{bbox[3].item()},"
                        f"1,-1,-1,-1\n"
                    )
            if not is_fake:
                os.makedirs(os.path.join(outputs_dir, "tracker"), exist_ok=True)
                with open(os.path.join(outputs_dir, "tracker", f"{sequence_name}.txt"), "w") as submit_file:
                    submit_file.writelines(sequence_tracker_results)
                logger.success(f"Submit sequence {sequence_name} done, FPS: {sequence_fps:.2f}. "
                               f"Saved to {os.path.join(outputs_dir, 'tracker', f'{sequence_name}.txt')}.",
                               only_main=False)
            else:
                logger.success(f"Fake submit sequence {sequence_name} done, FPS: {sequence_fps:.2f}.", only_main=False)
            pass
        else:
            raise NotImplementedError(f"Do not support to submit the results for dataset '{dataset}'.")

    # Post-process for submitting and evaluation:
    accelerator.wait_for_everyone()
    if state.num_processes == 1:
        timed = sum(x["timed_frames"] for x in performance)
        seconds = sum(x["timed_frames"] / x["fps"] for x in performance)
        os.makedirs(outputs_dir, exist_ok=True)
        with open(os.path.join(outputs_dir, "performance.json"), "w") as handle:
            json.dump({"sequences": performance, "FPS": timed / seconds if seconds else 0,
                       "timing": "wall time including loader, H2D, tracking; excludes first 10 frames per sequence; CUDA synchronized",
                       "peak_process_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}, handle, indent=2)
    if not is_evaluate:
        logger.success(
            log=f"Submit done. Saved to {os.path.join(outputs_dir, 'tracker')}",
            only_main=True,
        )
        return None
    else:
        if accelerator.is_main_process:
            logger.info(
                log=f"Start evaluation...",
                only_main=True,
            )
            # Prepare for evaluation:
            if dataset in ["DanceTrack", "SportsMOT", "MOT17", "BFT"]:
                gt_dir = os.path.join(data_root, dataset, data_split)
                tracker_dir = os.path.join(outputs_dir, "tracker")
            elif dataset in ["PersonPath22_Inference"]:
                gt_dir = os.path.join(data_root, dataset, "gts", "person_path_22-test")
                tracker_dir = os.path.join(outputs_dir, "tracker")
            else:
                raise NotImplementedError(f"Do not support to find the gt_dir for dataset '{dataset}'.")
            if dataset in ["DanceTrack", "SportsMOT", "BFT"] or (dataset in ["MOT17"] and data_split == "test"):
                args = {
                    "--SPLIT_TO_EVAL": data_split,
                    "--METRICS": ["HOTA", "CLEAR", "Identity"],
                    "--GT_FOLDER": gt_dir,
                    "--SEQMAP_FILE": str(subset_seqmap_path or os.path.join(data_root, dataset, f"{data_split}_seqmap.txt")),
                    "--SKIP_SPLIT_FOL": "True",
                    "--TRACKERS_TO_EVAL": "",
                    "--TRACKER_SUB_FOLDER": "",
                    "--USE_PARALLEL": "True",
                    "--NUM_PARALLEL_CORES": "8",
                    "--PLOT_CURVES": "False",
                    "--TRACKERS_FOLDER": tracker_dir,
                }
                cmd = [sys.executable, "TrackEval/scripts/run_mot_challenge.py"]
            elif dataset in ["PersonPath22_Inference"]:
                args = {
                    "--SPLIT_TO_EVAL": data_split,
                    "--METRICS": ["HOTA", "CLEAR", "Identity"],
                    "--GT_FOLDER": gt_dir,
                    "--USE_PARALLEL": "True",
                    "--NUM_PARALLEL_CORES": "8",
                    "--TRACKERS_FOLDER": tracker_dir,
                    "--BENCHMARK": "person_path_22",
                    "--SEQMAP_FILE": os.path.join(data_root, dataset, "gts", "seqmaps", "person_path_22-test.txt"),
                    "--SKIP_SPLIT_FOL": "True",
                    "--TRACKER_SUB_FOLDER": "",
                    "--TRACKERS_TO_EVAL": "",
                }
                cmd = ["python", "TrackEval/scripts/run_person_path_22.py"]
            else:
                raise NotImplementedError(
                    f"Do not support to eval the results for dataset '{dataset}' split '{data_split}'."
                )
            for k, v in args.items():
                cmd.append(k)
                if isinstance(v, list):
                    cmd += v
                else:
                    cmd.append(v)
            # Run the eval script:
            _ = subprocess.run(
                cmd,
            )
            # Check if the eval script is done:
            if _.returncode == 0:
                logger.success("Evaluation script is done.", only_main=True)
            else:
                raise RuntimeError("Evaluation script failed.")
        # Wait for all processes:
        accelerator.wait_for_everyone()
        # Get the metrics:
        eval_metrics_path = os.path.join(outputs_dir, "tracker", "pedestrian_summary.txt")
        eval_metrics_dict = get_eval_metrics_dict(metric_path=eval_metrics_path)
        metrics = Metrics()
        metrics["HOTA"].update(eval_metrics_dict["HOTA"])
        metrics["DetA"].update(eval_metrics_dict["DetA"])
        metrics["AssA"].update(eval_metrics_dict["AssA"])
        metrics["DetPr"].update(eval_metrics_dict["DetPr"])
        metrics["DetRe"].update(eval_metrics_dict["DetRe"])
        metrics["AssPr"].update(eval_metrics_dict["AssPr"])
        metrics["AssRe"].update(eval_metrics_dict["AssRe"])
        metrics["MOTA"].update(eval_metrics_dict["MOTA"])
        metrics["IDF1"].update(eval_metrics_dict["IDF1"])
        metrics["IDSW"].update(eval_metrics_dict["IDSW"])
        logger.success(
            log=f"Get evaluation metrics from {eval_metrics_path}.",
            only_main=True,
        )

        return metrics


@torch.no_grad()
def get_results_of_one_sequence(
        logger: Logger,
        runtime_tracker: RuntimeTracker,
        sequence_loader: DataLoader,
):
    tracker_results = []
    assert len(sequence_loader) > 10, "The sequence loader is too short."
    for t, (image, image_path) in enumerate(sequence_loader):
        if t == 10:
            torch.cuda.synchronize()
            begin_time = time.time()
        image.tensors = image.tensors.cuda()
        image.mask = image.mask.cuda()
        # image = nested_tensor_from_tensor_list(tensor_list=[image[0]])
        runtime_tracker.update(image=image)
        _results = runtime_tracker.get_track_results()
        tracker_results.append(_results)
    torch.cuda.synchronize()
    fps = (len(sequence_loader) - 10) / (time.time() - begin_time)
    return tracker_results, fps


def get_eval_metrics_dict(metric_path: str):
    with open(metric_path) as f:
        metric_names = f.readline()[:-1].split(" ")
        metric_values = f.readline()[:-1].split(" ")
    metrics = {
        n: float(v) for n, v in zip(metric_names, metric_values)
    }
    return metrics


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Get runtime option:
    opt = runtime_option()
    cfg = yaml_to_dict(opt.config_path)

    # Loading super config:
    if opt.super_config_path is not None:  # the runtime option is priority
        cfg = load_super_config(cfg, opt.super_config_path)
    else:  # if not, use the default super config path in the config file
        cfg = load_super_config(cfg, cfg["SUPER_CONFIG_PATH"])

    # Combine the config and runtime into config dict:
    cfg = update_config(config=cfg, option=opt)

    # Call the "train_engine" function:
    submit_and_evaluate(config=cfg)
