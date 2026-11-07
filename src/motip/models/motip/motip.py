# Copyright (c) Ruopeng Gao. All Rights Reserved.

import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class MOTIP(nn.Module):
    def __init__(
            self,
            detr: nn.Module,
            detr_framework: str,
            only_detr: bool,
            trajectory_modeling: nn.Module,
            id_decoder: nn.Module,
    ):
        super().__init__()
        self.detr = detr
        self.detr_framework = detr_framework
        self.only_detr = only_detr
        self.trajectory_modeling = trajectory_modeling
        self.id_decoder = id_decoder
        self.visual_trail = None
        self.freeze_detr = False

        if self.id_decoder is not None:
            self.num_id_vocabulary = self.id_decoder.num_id_vocabulary
        else:
            self.num_id_vocabulary = 1000           # hack implementation

        return

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_detr:
            self.detr.eval()
        return self

    def forward(self, **kwargs):
        assert "part" in kwargs, "Parameter `part` is required for MOTIP forward."
        match kwargs["part"]:
            case "detr":
                frames = kwargs["frames"]
                if "use_checkpoint" in kwargs:
                    return checkpoint(
                        self.detr, frames,
                        use_reentrant=False,
                    )
                else:
                    return self.detr(samples=frames)
            case "trajectory_modeling":
                seq_info = kwargs["seq_info"]
                if self.visual_trail is not None and "trajectory_rois" in seq_info:
                    seq_info = self.visual_trail(seq_info)
                return self.trajectory_modeling(seq_info)
            case "trail_extract":
                outputs = kwargs["detr_outputs"]
                return self.visual_trail.extract(outputs["trail_source"], outputs["pred_boxes"],
                                                 outputs["trail_valid_ratios"])
            case "id_decoder":
                seq_info = kwargs["seq_info"]
                use_decoder_checkpoint = kwargs["use_decoder_checkpoint"] if "use_decoder_checkpoint" in kwargs else False
                return self.id_decoder(seq_info, use_decoder_checkpoint=use_decoder_checkpoint)
            case _:
                raise NotImplementedError(f"MOTIP forwarding doesn't support part={kwargs['part']}.")
