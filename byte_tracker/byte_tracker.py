#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

import numpy as np

from byte_tracker.tracker.byte_tracker import BYTETracker


class ByteTracker(object):
    def __init__(self, args, frame_rate):
        self.args = args

        # ByteTrackerインスタンス生成
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def __call__(
        self,
        detections,
        frame,
        input_shape,
    ):
        # トラッカー更新
        bboxes, ids, scores = self._tracker_update(
            detections,
            frame,
            input_shape,
        )

        return bboxes, ids, scores

    def _tracker_update(self, dets, image, input_shape):
        image_info = {'id': 0}
        image_info['image'] = copy.deepcopy(image)
        image_info['width'] = image.shape[1]
        image_info['height'] = image.shape[0]
        ratio = min(input_shape[0] / image.shape[0],
                    input_shape[1] / image.shape[1])
        image_info['ratio'] = ratio

        # トラッカー更新
        online_targets = []
        if dets is not None:
            online_targets = self.tracker.update(
                dets[:, :-1],
                [image_info['height'], image_info['width']],
                [image_info['height'], image_info['width']],
            )

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            if tlwh[2] * tlwh[3] > self.args.min_box_area:
                online_tlwhs.append(
                    np.array([
                        tlwh[0], tlwh[1], tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]
                    ]))
                online_ids.append(track_id)
                online_scores.append(online_target.score)

        return online_tlwhs, online_scores, online_ids
