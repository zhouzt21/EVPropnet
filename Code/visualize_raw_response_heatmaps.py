#!/usr/bin/env python3

import argparse
import importlib
import os
import sys
import types

import cv2
import numpy as np
import tensorflow as tf

import Misc.ImageUtils as iu
import Misc.TFUtils as tu
from test_raw_ckpt import build_event_frame_classic
from test_raw_ckpt import build_event_frame_gray
from test_raw_ckpt import merge_boxes
from test_raw_ckpt import prepare_input


tf.compat.v1.disable_eager_execution()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', required=True, help='Path to input RAW file')
    parser.add_argument('--ckpt_path', required=True, help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--delta_t_us', type=int, default=50000, help='Frame accumulation time in microseconds')
    parser.add_argument('--start_ts_us', type=int, default=0, help='RAW start timestamp in microseconds')
    parser.add_argument('--max_duration_us', type=int, default=None, help='Maximum RAW duration to process in microseconds')
    parser.add_argument('--num_frames', type=int, default=6, help='Number of preview frames to save')
    parser.add_argument('--skip_frames', type=int, default=0, help='Number of accumulated frames to skip before saving')
    parser.add_argument('--save_every', type=int, default=1, help='Save every Nth frame after skip_frames')
    parser.add_argument('--width', type=int, default=1280, help='Sensor width')
    parser.add_argument('--height', type=int, default=720, help='Sensor height')
    parser.add_argument('--network', default='Network.ResNet', help='Network module, e.g., Network.ResNet')
    parser.add_argument('--init_neurons', type=float, default=32, help='Init neurons')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device, -1 for CPU')
    parser.add_argument('--merge_dist', type=int, default=100, help='Merge distance for bbox clustering')
    parser.add_argument('--min_area', type=int, default=5, help='Minimum connected-component area')
    parser.add_argument('--overlay_threshold', type=float, default=0.03, help='Minimum clipped response shown in overlay')
    parser.add_argument('--overlay_alpha', type=float, default=0.72, help='Heatmap opacity in overlay')
    parser.add_argument('--crop_size', type=int, default=160, help='Square zoom crop size in pixels')
    parser.add_argument('--zoom_scale', type=int, default=3, help='Integer scale factor for zoom panels')
    parser.add_argument('--vis_style', default='prophesee', choices=['prophesee', 'classic'],
                        help='Event-frame visualization style')
    parser.add_argument('--save_npz', type=int, default=1, help='1: save response arrays as npz')
    return parser.parse_args()


def import_events_iterator():
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

        from metavision_core.event_io import EventsIterator
    except ImportError as exc:
        raise ImportError('metavision_core is required to read RAW files. Please install Metavision SDK Python bindings first.') from exc
    return EventsIterator


def response_to_uint8(response):
    clipped = np.clip(response, 0.0, 1.0)
    return (clipped * 255.0).astype(np.uint8), clipped


def response_to_regions(response_u8, merge_dist, min_area):
    _, binary = cv2.threshold(response_u8, 0, 255, cv2.THRESH_BINARY)
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary)

    regions = []
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            regions.append([x, y, w, h])
    return merge_boxes(regions, merge_dist)


def build_event_display(events, height, width, vis_style):
    if vis_style == 'classic':
        return build_event_frame_classic(events, height, width)
    return iu.PropheseeEventsDisp(events, width=width, height=height)


def put_label(image, text, origin=(14, 32)):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def draw_regions(image, regions):
    for x, y, w, h in regions:
        cv2.rectangle(image, (x, y), (x + w, y + h), (80, 255, 80), 2)
    return image


def make_overlay(event_disp, heatmap, response_clipped, threshold, alpha):
    overlay = event_disp.copy()
    mask = response_clipped >= threshold
    if np.any(mask):
        blended = cv2.addWeighted(event_disp, 1.0 - alpha, heatmap, alpha, 0.0)
        overlay[mask] = blended[mask]
    return overlay


def make_contact_sheet(rows):
    if not rows:
        return None
    row_images = [np.hstack(row) for row in rows]
    return np.vstack(row_images)


def find_zoom_center(response_clipped, event_gray):
    if float(np.max(response_clipped)) > 0.0:
        y, x = np.unravel_index(np.argmax(response_clipped), response_clipped.shape)
        return int(x), int(y)

    activity = np.abs(event_gray.astype(np.float32) - 127.0)
    activity = cv2.GaussianBlur(activity, (0, 0), sigmaX=8.0, sigmaY=8.0)
    y, x = np.unravel_index(np.argmax(activity), activity.shape)
    return int(x), int(y)


def crop_square(image, center_xy, crop_size):
    x, y = center_xy
    h, w = image.shape[:2]
    half = crop_size // 2
    x1 = max(0, min(w - crop_size, x - half))
    y1 = max(0, min(h - crop_size, y - half))
    x2 = min(w, x1 + crop_size)
    y2 = min(h, y1 + crop_size)
    crop = image[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def resize_zoom(image, scale):
    if scale <= 1:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def local_response_heatmap(response_crop):
    max_val = float(np.max(response_crop))
    if max_val <= 0.0:
        local_u8 = np.zeros_like(response_crop, dtype=np.uint8)
    else:
        local_u8 = np.clip(response_crop / max_val, 0.0, 1.0)
        local_u8 = (local_u8 * 255.0).astype(np.uint8)
    return cv2.applyColorMap(local_u8, cv2.COLORMAP_INFERNO)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    EventsIterator = import_events_iterator()
    args.net = importlib.import_module(args.network)
    tu.SetGPU(args.gpu)

    img_size = np.array([args.height, args.width, 1])
    input_ph = tf.compat.v1.placeholder(tf.float32, shape=(1, img_size[0], img_size[1], img_size[2]), name='Input')

    class_name = args.network.replace('Network.', '').split('Net')[0] + 'Net'
    network_cls = getattr(args.net, class_name)
    vn = network_cls(InputPH=input_ph, InitNeurons=args.init_neurons, Suffix='')
    pr_val = vn.Network()
    saver = tf.compat.v1.train.Saver()

    iterator_kwargs = {
        'input_path': args.raw_path,
        'start_ts': args.start_ts_us,
        'delta_t': args.delta_t_us,
    }
    if args.max_duration_us is not None:
        iterator_kwargs['max_duration'] = args.max_duration_us

    base_name = os.path.splitext(os.path.basename(args.raw_path))[0]
    summary_path = os.path.join(args.output_dir, '{}_response_summary.txt'.format(base_name))
    sheet_rows = []
    zoom_rows = []
    saved = 0

    with tf.compat.v1.Session() as sess, open(summary_path, 'w', encoding='utf-8') as summary:
        saver.restore(sess, args.ckpt_path)
        summary.write('raw_path {}\n'.format(args.raw_path))
        summary.write('ckpt_path {}\n'.format(args.ckpt_path))
        summary.write('timestamp_us max_response mean_response positive_pixels regions\n')

        iterator = EventsIterator(**iterator_kwargs)
        current_timestamp_us = int(args.start_ts_us)
        frame_index = 0

        for events in iterator:
            current_timestamp_us += args.delta_t_us
            frame_index += 1
            if frame_index <= args.skip_frames:
                continue
            if (frame_index - args.skip_frames - 1) % args.save_every != 0:
                continue

            event_gray = build_event_frame_gray(events, args.height, args.width)
            ibatch = prepare_input(event_gray, img_size)
            response = np.squeeze(sess.run([pr_val], feed_dict={vn.InputPH: ibatch}))
            response_u8, response_clipped = response_to_uint8(response)
            regions = response_to_regions(response_u8, args.merge_dist, args.min_area)

            event_disp = build_event_display(events, args.height, args.width, args.vis_style)
            heatmap = cv2.applyColorMap(response_u8, cv2.COLORMAP_INFERNO)
            overlay = make_overlay(event_disp, heatmap, response_clipped, args.overlay_threshold, args.overlay_alpha)
            overlay = draw_regions(overlay, regions)
            zoom_center = find_zoom_center(response_clipped, event_gray)

            label = '{} t={:.3f}s max={:.3f} regions={}'.format(
                base_name, current_timestamp_us / 1000000.0, float(np.max(response_clipped)), len(regions))
            event_panel = put_label(event_disp.copy(), 'events')
            heat_panel = put_label(heatmap.copy(), 'network response')
            overlay_panel = put_label(overlay.copy(), label)

            prefix = '{}_{:09d}'.format(base_name, current_timestamp_us)
            cv2.imwrite(os.path.join(args.output_dir, prefix + '_events.png'), event_panel)
            cv2.imwrite(os.path.join(args.output_dir, prefix + '_response_heatmap.png'), heat_panel)
            cv2.imwrite(os.path.join(args.output_dir, prefix + '_response_overlay.png'), overlay_panel)

            event_crop, crop_box = crop_square(event_disp, zoom_center, args.crop_size)
            response_crop, _ = crop_square(response_clipped, zoom_center, args.crop_size)
            heat_crop, _ = crop_square(heatmap, zoom_center, args.crop_size)
            overlay_crop, _ = crop_square(overlay, zoom_center, args.crop_size)
            local_heat_crop = local_response_heatmap(response_crop)

            zoom_label = 'zoom x={} y={} max={:.3f}'.format(zoom_center[0], zoom_center[1], float(np.max(response_crop)))
            event_zoom = put_label(resize_zoom(event_crop.copy(), args.zoom_scale), 'events zoom')
            abs_zoom = put_label(resize_zoom(heat_crop.copy(), args.zoom_scale), 'absolute response')
            local_zoom = put_label(resize_zoom(local_heat_crop.copy(), args.zoom_scale), 'local response')
            overlay_zoom = put_label(resize_zoom(overlay_crop.copy(), args.zoom_scale), zoom_label)
            cv2.imwrite(os.path.join(args.output_dir, prefix + '_zoom_events.png'), event_zoom)
            cv2.imwrite(os.path.join(args.output_dir, prefix + '_zoom_response_abs.png'), abs_zoom)
            cv2.imwrite(os.path.join(args.output_dir, prefix + '_zoom_response_local.png'), local_zoom)
            cv2.imwrite(os.path.join(args.output_dir, prefix + '_zoom_overlay.png'), overlay_zoom)

            if args.save_npz:
                np.savez_compressed(
                    os.path.join(args.output_dir, prefix + '_response.npz'),
                    response=response.astype(np.float32),
                    response_clipped=response_clipped.astype(np.float32),
                    timestamp_us=np.array([current_timestamp_us], dtype=np.int64),
                    crop_box=np.array(crop_box, dtype=np.int32),
                )

            positive_pixels = int(np.count_nonzero(response_u8))
            summary.write('{} {:.6f} {:.6f} {} {}\n'.format(
                current_timestamp_us,
                float(np.max(response_clipped)),
                float(np.mean(response_clipped)),
                positive_pixels,
                len(regions),
            ))

            sheet_rows.append([event_panel, heat_panel, overlay_panel])
            zoom_rows.append([event_zoom, abs_zoom, local_zoom, overlay_zoom])
            saved += 1
            print('Saved', prefix, 'max_response', float(np.max(response_clipped)), 'regions', len(regions))
            if saved >= args.num_frames:
                break

    sheet = make_contact_sheet(sheet_rows)
    if sheet is not None:
        cv2.imwrite(os.path.join(args.output_dir, '{}_contact_sheet.png'.format(base_name)), sheet)
    zoom_sheet = make_contact_sheet(zoom_rows)
    if zoom_sheet is not None:
        cv2.imwrite(os.path.join(args.output_dir, '{}_zoom_contact_sheet.png'.format(base_name)), zoom_sheet)
    print('Done. Saved frames:', saved)
    print('Output dir:', args.output_dir)


if __name__ == '__main__':
    main()
