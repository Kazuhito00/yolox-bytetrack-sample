#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2
import numpy as np

from yolox.yolox_onnx import YoloxONNX
from byte_tracker.byte_tracker import ByteTracker


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    # YOLOX parameters
    parser.add_argument(
        "--yolox_model",
        type=str,
        default='model/yolox_nano.onnx',
    )
    parser.add_argument(
        '--input_shape',
        type=str,
        default="416,416",
        help="Specify an input shape for inference.",
    )
    parser.add_argument(
        '--score_th',
        type=float,
        default=0.3,
        help='Class confidence',
    )
    parser.add_argument(
        '--nms_th',
        type=float,
        default=0.45,
        help='NMS IoU threshold',
    )
    parser.add_argument(
        '--nms_score_th',
        type=float,
        default=0.1,
        help='NMS Score threshold',
    )
    parser.add_argument(
        "--with_p6",
        action="store_true",
        help="Whether your model uses p6 in FPN/PAN.",
    )

    # motpy parameters
    parser.add_argument(
        "--track_thresh",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--track_buffer",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--match_thresh",
        type=float,
        default=0.8,
    )
    parser.add_argument(
        "--min_box_area",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--mot20",
        action="store_true",
    )

    args = parser.parse_args()

    return args


class dict_dot_notation(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def main():
    # 引数解析 #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    if args.movie is not None:
        cap_device = args.movie

    # YOLOX parameters
    model_path = args.yolox_model
    input_shape = tuple(map(int, args.input_shape.split(',')))
    score_th = args.score_th
    nms_th = args.nms_th
    nms_score_th = args.nms_score_th
    with_p6 = args.with_p6

    # ByteTrack parameters
    track_thresh = args.track_thresh
    track_buffer = args.track_buffer
    match_thresh = args.match_thresh
    min_box_area = args.min_box_area
    mot20 = args.mot20

    # カメラ準備 ###############################################################
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)

    # モデルロード #############################################################
    yolox = YoloxONNX(
        model_path=model_path,
        input_shape=input_shape,
        class_score_th=score_th,
        nms_th=nms_th,
        nms_score_th=nms_score_th,
        with_p6=with_p6,
        providers=['CPUExecutionProvider'],
    )

    tracker = ByteTracker(
        args=dict_dot_notation({
            'track_thresh': track_thresh,
            'track_buffer': track_buffer,
            'match_thresh': match_thresh,
            'min_box_area': min_box_area,
            'mot20': mot20,
        }),
        frame_rate=int(cap_fps + 0.49),
    )

    # トラッキングID保持用変数
    track_id_dict = {}

    while True:
        start_time = time.time()

        # カメラキャプチャ ################################################
        ret, frame = cap.read()
        if not ret:
            break
        debug_image = copy.deepcopy(frame)

        # 推論実施 ########################################################
        # Object Detection
        bboxes, scores, class_ids = yolox.inference(frame)
        detections = [[*b, s, l] for b, s, l in zip(bboxes, scores, class_ids)]
        detections = np.array(detections)

        # Multi Object Tracking
        bboxes, scores, trakcer_ids = tracker(
            detections,
            frame,
            input_shape,
        )

        # トラッキングIDと連番の紐付け
        for trakcer_id, bbox in zip(trakcer_ids, bboxes):
            if trakcer_id not in track_id_dict:
                new_id = len(track_id_dict)
                track_id_dict[trakcer_id] = new_id

        elapsed_time = time.time() - start_time

        # デバッグ描画
        debug_image = draw_debug(
            debug_image,
            elapsed_time,
            score_th,
            bboxes,
            scores,
            trakcer_ids,
            track_id_dict,
        )

        # キー処理(ESC：終了) ##############################################
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # 画面反映 #########################################################
        cv2.imshow('YOLOX ByteTrack Sample', debug_image)

    cap.release()
    cv2.destroyAllWindows()


def get_id_color(index):
    temp_index = abs(int(index)) * 3
    color = ((37 * temp_index) % 255, (17 * temp_index) % 255,
             (29 * temp_index) % 255)
    return color


def draw_debug(
    image,
    elapsed_time,
    score_th,
    bboxes,
    scores,
    trakcer_ids,
    track_id_dict,
):
    debug_image = copy.deepcopy(image)

    for bbox, score, tracker_id in zip(bboxes, scores, trakcer_ids):
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        if score_th > score:
            continue

        color = get_id_color(int(track_id_dict[tracker_id]))

        # バウンディングボックス
        debug_image = cv2.rectangle(
            debug_image,
            (x1, y1),
            (x2, y2),
            color,
            thickness=2,
        )

        # クラスID、スコア
        text = '%s' % (str(tracker_id))
        debug_image = cv2.putText(
            debug_image,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness=2,
        )

    # 推論時間
    text = 'Elapsed time:' + '%.0f' % (elapsed_time * 1000)
    text = text + 'ms'
    debug_image = cv2.putText(
        debug_image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        thickness=2,
    )

    return debug_image


if __name__ == '__main__':
    main()
