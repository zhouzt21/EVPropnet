#!/usr/bin/env python3

import os
import argparse
import numpy as np
import cv2
import tensorflow as tf
import importlib
import types
import sys

import Misc.TFUtils as tu
import Misc.ImageUtils as iu
from response_postprocess import (
    extract_scored_regions,
    input_transform,
    map_bbox_to_sensor,
    load_window_starts_us,
    scheduled_window_bounds,
    window_timestamp_us,
)
from bad_pixel_filter import filter_events, load_bad_pixel_mask

tf.compat.v1.disable_eager_execution()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', required=True, help='Path to input RAW file')
    parser.add_argument('--ckpt_path', required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--delta_t_us', type=int, default=50000, help='Frame accumulation time in microseconds')
    parser.add_argument('--start_ts_us', type=int, default=0, help='RAW start timestamp in microseconds')
    parser.add_argument('--max_duration_us', type=int, default=None, help='Maximum RAW duration to process in microseconds')
    parser.add_argument('--window_starts_file', default=None,
                        help='Optional text file with explicit absolute window starts, one per line')
    parser.add_argument('--width', type=int, default=1280, help='Sensor width')
    parser.add_argument('--height', type=int, default=720, help='Sensor height')
    parser.add_argument('--network_width', type=int, default=0,
                        help='Network input width; 0 uses sensor width')
    parser.add_argument('--network_height', type=int, default=0,
                        help='Network input height; 0 uses sensor height')
    parser.add_argument('--input_transform', choices=['center_crop', 'stretch'], default='center_crop',
                        help='How the sensor frame is mapped to the network input')
    parser.add_argument('--input_interpolation', choices=['linear', 'nearest'], default='linear',
                        help='Resize interpolation used by stretch; match the training cache')
    parser.add_argument('--network', default='Network.ResNet', help='Network module, e.g., Network.ResNet')
    parser.add_argument('--init_neurons', type=float, default=32, help='Init neurons')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device, -1 for CPU')
    parser.add_argument('--merge_dist', type=int, default=100, help='Merge distance for bbox clustering')
    parser.add_argument('--response_threshold', type=float, default=0.0,
                        help='Native response threshold in [0, 1]')
    parser.add_argument('--response_activation', choices=['clipped', 'sigmoid'], default='clipped',
                        help='How raw network output is converted to the response map')
    parser.add_argument('--min_component_area', type=int, default=5,
                        help='Minimum connected-component area in response pixels')
    parser.add_argument('--timestamp_anchor', choices=['start', 'center', 'end'], default='end',
                        help='Timestamp written for each accumulation window')
    parser.add_argument('--save_video', type=int, default=1, help='1: save mp4, 0: disable')
    parser.add_argument('--fps', type=float, default=20.0, help='Output video fps')
    parser.add_argument('--vis_style', default='prophesee', choices=['prophesee', 'classic'],
                        help='Visualization style for video: prophesee or classic')
    parser.add_argument('--bad_pixel_mask', default=None,
                        help='Versioned JSON bad-pixel mask applied before frame construction')
    return parser.parse_args()


def merge_boxes(regions, tau):
    merged = []
    used = [False] * len(regions)

    for i in range(len(regions)):
        if used[i]:
            continue

        x1, y1, w1, h1 = regions[i]
        x_min, y_min = x1, y1
        x_max, y_max = x1 + w1, y1 + h1

        used[i] = True

        for j in range(i + 1, len(regions)):
            if used[j]:
                continue

            x2, y2, w2, h2 = regions[j]

            dx = max(x1 - (x2 + w2), x2 - (x1 + w1), 0)
            dy = max(y1 - (y2 + h2), y2 - (y1 + h1), 0)

            if dx * dx + dy * dy <= tau * tau:
                used[j] = True
                x_min = min(x_min, x2)
                y_min = min(y_min, y2)
                x_max = max(x_max, x2 + w2)
                y_max = max(y_max, y2 + h2)

        merged.append([x_min, y_min, x_max - x_min, y_max - y_min])

    return merged


def build_event_frame_gray(events, height, width):
    frame = np.full((height, width), 127, dtype=np.uint8)
    if events.size == 0:
        return frame

    xs = events['x'].astype(np.int64)
    ys = events['y'].astype(np.int64)
    ps = events['p'].astype(np.bool_)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[valid]
    ys = ys[valid]
    ps = ps[valid]

    if xs.size == 0:
        return frame

    values = np.empty(xs.size, dtype=np.uint8)
    values[ps] = 255
    values[~ps] = 0
    frame[ys, xs] = values
    return frame


def build_event_frame_classic(events, height, width):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if events.size == 0:
        return frame

    xs = events['x'].astype(np.int64)
    ys = events['y'].astype(np.int64)
    ps = events['p'].astype(np.bool_)

    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    xs = xs[valid]
    ys = ys[valid]
    ps = ps[valid]

    if xs.size == 0:
        return frame

    colors = np.zeros((xs.size, 3), dtype=np.uint8)
    colors[ps, 2] = 255
    colors[~ps, 0] = 255
    frame[ys, xs] = colors
    return frame


def prepare_input(frame, target_shape, mode='center_crop', interpolation=cv2.INTER_LINEAR):
    if frame.ndim == 2:
        frame = frame[:, :, np.newaxis]

    if mode == 'center_crop':
        cropped = iu.CenterCrop(frame, target_shape)
        if cropped is None:
            raise ValueError('Network input is larger than sensor frame for center_crop')
        frame = cropped
    elif mode == 'stretch':
        frame = cv2.resize(frame, (target_shape[1], target_shape[0]), interpolation=interpolation)
        frame = frame[:, :, np.newaxis]
    else:
        raise ValueError('Unknown input transform: {}'.format(mode))

    ibatch = np.sign(np.float32(frame) - 127.0)
    ibatch = ibatch[np.newaxis, :, :, :]
    return ibatch


def main():
    args = parse_args()
    bad_pixel_mask = (
        load_bad_pixel_mask(args.bad_pixel_mask) if args.bad_pixel_mask else None
    )

    try:
        try:
            import h5py  # noqa: F401
        except Exception:
            h5py_stub = types.ModuleType('h5py')

            class _H5Stub(object):
                def __init__(self, *args, **kwargs):
                    raise ImportError('h5py is unavailable in this environment')

            h5py_stub.File = _H5Stub
            h5py_stub.__version__ = 'stub'
            sys.modules['h5py'] = h5py_stub

        from metavision_core.event_io import RawReader
    except ImportError as exc:
        raise ImportError('metavision_core is required to read RAW files. Please install Metavision SDK Python bindings first.') from exc

    os.makedirs(args.output_dir, exist_ok=True)
    bbox_path = os.path.join(args.output_dir, 'BBoxes.txt')

    if args.save_video:
        accum_ms = args.delta_t_us / 1000.0
        if accum_ms.is_integer():
            accum_tag = '{}ms'.format(int(accum_ms))
        else:
            accum_tag = '{:.2f}ms'.format(accum_ms).replace('.', 'p')
        video_path = os.path.join(args.output_dir, 'pred_{}.mp4'.format(accum_tag))
    else:
        video_path = None

    args.net = importlib.import_module(args.network)
    tu.SetGPU(args.gpu)

    network_height = int(args.network_height) if int(args.network_height) > 0 else int(args.height)
    network_width = int(args.network_width) if int(args.network_width) > 0 else int(args.width)
    img_size = np.array([network_height, network_width, 1])
    coordinate_transform = input_transform(
        sensor_height=args.height,
        sensor_width=args.width,
        network_height=network_height,
        network_width=network_width,
        mode=args.input_transform,
    )

    input_ph = tf.compat.v1.placeholder(tf.float32, shape=(1, img_size[0], img_size[1], img_size[2]), name='Input')

    class_name = args.network.replace('Network.', '').split('Net')[0] + 'Net'
    network_cls = getattr(args.net, class_name)
    vn = network_cls(InputPH=input_ph, InitNeurons=args.init_neurons, Suffix='')
    pr_val = vn.Network()

    saver = tf.compat.v1.train.Saver()

    writer = None
    if video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, args.fps, (args.width, args.height))
        if not writer.isOpened():
            raise RuntimeError('Failed to open VideoWriter: {}'.format(video_path))

    with tf.compat.v1.Session() as sess, open(bbox_path, 'w') as bbox_logs:
        bbox_logs.write('FileName x y w h response_max response_mean response_area\n')
        saver.restore(sess, args.ckpt_path)
        print('Loaded checkpoints:', args.ckpt_path)

        reader = RawReader(str(args.raw_path))
        schedule_duration_us = (
            int(args.max_duration_us)
            if args.max_duration_us is not None
            else (2 ** 62)
        )
        written_frames = 0

        if args.window_starts_file:
            window_schedule = (
                (int(window_start_us), int(window_start_us) + int(args.delta_t_us))
                for window_start_us in load_window_starts_us(args.window_starts_file)
            )
        else:
            window_schedule = scheduled_window_bounds(
                args.start_ts_us, args.delta_t_us, schedule_duration_us
            )
        for window_start_us, current_timestamp_us in window_schedule:
            if reader.is_done():
                break
            if int(reader.current_time) != int(window_start_us):
                reader.seek_time(int(window_start_us))
            events = reader.load_delta_t(int(args.delta_t_us))
            if bad_pixel_mask is not None:
                events = filter_events(
                    events,
                    bad_pixel_mask['pixel_coordinates'],
                    width=args.width,
                )

            frame = build_event_frame_gray(events, args.height, args.width)
            interpolation = cv2.INTER_NEAREST if args.input_interpolation == 'nearest' else cv2.INTER_LINEAR
            ibatch = prepare_input(frame, img_size, mode=args.input_transform, interpolation=interpolation)

            pr_val_ret = np.squeeze(sess.run([pr_val], feed_dict={vn.InputPH: ibatch}))
            if args.response_activation == 'sigmoid':
                response = (1.0 / (1.0 + np.exp(-np.clip(pr_val_ret, -40.0, 40.0)))).astype(np.float32)
            else:
                response = np.clip(pr_val_ret, 0.0, 1.0).astype(np.float32)
            regions = extract_scored_regions(
                response,
                response_threshold=args.response_threshold,
                min_component_area=args.min_component_area,
                merge_dist=args.merge_dist,
            )
            for region in regions:
                region['bbox_xywh'] = map_bbox_to_sensor(
                    region['bbox_xywh'], coordinate_transform)

            output_timestamp_us = window_timestamp_us(
                current_timestamp_us, args.delta_t_us, args.timestamp_anchor)
            file_name = '{}_{}.png'.format(
                os.path.splitext(os.path.basename(args.raw_path))[0], output_timestamp_us)
            if len(regions) == 0:
                bbox_logs.write('{} {} {} {} {} {} {} {}\n'.format(
                    file_name, None, None, None, None, None, None, None))
            else:
                for region in regions:
                    x, y, w, h = region['bbox_xywh']
                    bbox_logs.write('{} {} {} {} {} {:.9g} {:.9g} {}\n'.format(
                        file_name,
                        x,
                        y,
                        w,
                        h,
                        region['response_max'],
                        region['response_mean'],
                        region['response_area'],
                    ))

            if writer is not None:
                if args.vis_style == 'classic':
                    disp = build_event_frame_classic(events, args.height, args.width)
                else:
                    disp = iu.PropheseeEventsDisp(events, width=args.width, height=args.height)
                for region in regions:
                    x, y, w, h = region['bbox_xywh']
                    cv2.rectangle(
                        disp,
                        (int(round(x)), int(round(y))),
                        (int(round(x + w)), int(round(y + h))),
                        (0, 255, 0),
                        2,
                    )
                cv2.putText(disp, 't={:.3f}s'.format(current_timestamp_us / 1e6), (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                writer.write(disp)

            written_frames += 1
            if written_frames % 50 == 0:
                print('Processed frames:', written_frames, 'timestamp_us:', current_timestamp_us)

    if writer is not None:
        writer.release()

    print('Done. Total frames:', written_frames)


if __name__ == '__main__':
    main()
