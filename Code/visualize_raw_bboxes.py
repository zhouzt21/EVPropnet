#!/usr/bin/env python3

import os
import re
import cv2
import argparse
import numpy as np
import Misc.ImageUtils as iu


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', required=True, help='Path to input RAW file')
    parser.add_argument('--bboxes_path', required=True, help='Path to BBoxes.txt')
    parser.add_argument('--output_path', required=True, help='Path to output mp4 video')
    parser.add_argument('--width', type=int, default=1280, help='Sensor width')
    parser.add_argument('--height', type=int, default=720, help='Sensor height')
    parser.add_argument('--delta_t_us', type=int, default=10000, help='Frame accumulation time in microseconds')
    parser.add_argument('--fps', type=float, default=100.0, help='Output video fps')
    parser.add_argument('--vis_style', default='prophesee', choices=['prophesee', 'classic'],
                        help='Visualization style: prophesee or classic')
    return parser.parse_args()


def load_bboxes(bboxes_path):
    bbox_map = {}
    frame_pattern = re.compile(r'_(\d+)\.png$')

    with open(bboxes_path, 'r', encoding='utf-8') as f:
        header_skipped = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not header_skipped:
                header_skipped = True
                continue

            parts = line.split()
            if len(parts) != 5:
                continue

            file_name, x, y, w, h = parts
            match = frame_pattern.search(file_name)
            if not match:
                continue

            timestamp_us = int(match.group(1))
            bbox_map.setdefault(timestamp_us, [])

            if x == 'None':
                continue

            bbox_map[timestamp_us].append((int(x), int(y), int(w), int(h)))

    if not bbox_map:
        raise ValueError('No bbox records found in BBoxes.txt')

    timestamps = sorted(bbox_map.keys())
    print('Loaded bbox timestamps:', len(timestamps))
    print('BBox range:', timestamps[0], 'to', timestamps[-1], 'microseconds')
    return bbox_map, timestamps


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


def build_event_frame(events, height, width, vis_style):
    if vis_style == 'classic':
        return build_event_frame_classic(events, height, width)
    return iu.PropheseeEventsDisp(events, width=width, height=height)


def draw_bboxes(frame, boxes, timestamp_us):
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label = 't = {:.3f}s'.format(timestamp_us / 1000000.0)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'boxes = {}'.format(len(boxes)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    return frame


def main():
    args = parse_args()

    try:
        from metavision_core.event_io import EventsIterator
    except ImportError as exc:
        raise ImportError('metavision_core is required to read RAW files. Please install Metavision SDK Python bindings first.') from exc

    bbox_map, timestamps = load_bboxes(args.bboxes_path)
    max_timestamp_us = timestamps[-1]

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.output_path, fourcc, args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError('Failed to open VideoWriter: {}'.format(args.output_path))

    print('RAW path:', args.raw_path)
    print('Output path:', args.output_path)
    print('Frame size:', args.width, 'x', args.height)
    print('delta_t_us:', args.delta_t_us)
    print('fps:', args.fps)

    iterator = EventsIterator(input_path=args.raw_path, delta_t=args.delta_t_us)

    current_timestamp_us = 0
    written_frames = 0

    for events in iterator:
        current_timestamp_us += args.delta_t_us
        if current_timestamp_us > max_timestamp_us:
            break

        frame = build_event_frame(events, args.height, args.width, args.vis_style)
        boxes = bbox_map.get(current_timestamp_us, [])
        frame = draw_bboxes(frame, boxes, current_timestamp_us)
        writer.write(frame)
        written_frames += 1

        if written_frames % 100 == 0:
            print('Written frames:', written_frames, 'current_timestamp_us:', current_timestamp_us)

    writer.release()
    print('Done. Written frames:', written_frames)


if __name__ == '__main__':
    main()
