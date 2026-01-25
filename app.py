from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[1]
LANESIGHT_ROOT = REPO_ROOT / "LANESIGHT"

# Keep UI thin: do not modify the vision engine's imports/logic.
# The engine uses top-level imports like `from inference...`, so we add
# the LANESIGHT source root to sys.path to make those modules importable.
for p in (REPO_ROOT, LANESIGHT_ROOT):
	p_str = str(p)
	if p_str not in sys.path:
		sys.path.insert(0, p_str)


from LANESIGHT.video.infer_video import (  # noqa: E402
	FourMasks,
	VisionMaskGenerator,
	make_split_2x2,
	overlay_single_mask,
)


@dataclass(frozen=True)
class UiConfig:
	enable_vit: bool
	enable_ufld: bool
	enable_traditional: bool
	enable_yolo: bool

	out_fps: float
	preview_every_n_frames: int

	export_to_disk: bool
	export_path: Optional[Path]


def _write_uploaded_to_temp(uploaded_bytes: bytes, suffix: str) -> Path:
	fd, tmp_path = tempfile.mkstemp(suffix=suffix)
	try:
		with open(fd, "wb", closefd=False) as f:
			f.write(uploaded_bytes)
	finally:
		try:
			# Ensure the OS handle is released on Windows.
			import os
			os.close(fd)
		except Exception:
			pass
	return Path(tmp_path)


@st.cache_resource(show_spinner=True)
def get_engine() -> VisionMaskGenerator:
	# Instantiated once per Streamlit session: heavy weights/models live here.
	return VisionMaskGenerator()


def _get_video_props(cap: cv2.VideoCapture) -> Tuple[float, int, int]:
	in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
	return in_fps, width, height


def _selective_masks(masks: FourMasks, cfg: UiConfig) -> FourMasks:
	zeros = np.zeros_like(masks.vit_mask, dtype=np.uint8)
	return FourMasks(
		vit_mask=masks.vit_mask if cfg.enable_vit else zeros,
		ufld_mask=masks.ufld_mask if cfg.enable_ufld else zeros,
		traditional_mask=masks.traditional_mask if cfg.enable_traditional else zeros,
		yolo_mask=masks.yolo_mask if cfg.enable_yolo else zeros,
	)


def process_video(
	*,
	input_path: Path,
	cfg: UiConfig,
	engine: VisionMaskGenerator,
	progress: "st.delta_generator.DeltaGenerator",
	status: "st.delta_generator.DeltaGenerator",
	preview_slots: Tuple[
		"st.delta_generator.DeltaGenerator",
		"st.delta_generator.DeltaGenerator",
		"st.delta_generator.DeltaGenerator",
		"st.delta_generator.DeltaGenerator",
	],
) -> Path:
	cap = cv2.VideoCapture(str(input_path))
	if not cap.isOpened():
		raise RuntimeError(f"Could not open video: {input_path}")

	in_fps, width, height = _get_video_props(cap)
	if width <= 0 or height <= 0:
		cap.release()
		raise RuntimeError("Could not read video dimensions.")

	# Clamp requested FPS to input FPS for predictable export.
	out_fps = float(cfg.out_fps)
	if in_fps > 0:
		out_fps = min(out_fps, in_fps)

	frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	use_pos_msec = True
	if out_fps <= 0:
		# Should not happen due to UI constraints, but keep it robust.
		out_fps = 25.0

	sample_period_s = 1.0 / out_fps
	next_sample_t_s = 0.0

	# Output temp mp4
	out_fd, out_tmp = tempfile.mkstemp(suffix=".mp4")
	try:
		import os
		os.close(out_fd)
	except Exception:
		pass
	out_path = Path(out_tmp)

	# Write split 2x2 (half-res quadrants)
	w2, h2 = max(1, width // 2), max(1, height // 2)
	out_size = (w2 * 2, h2 * 2)

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(out_path), fourcc, out_fps if out_fps > 0 else 25.0, out_size)
	if not writer.isOpened():
		cap.release()
		raise RuntimeError("Could not open VideoWriter (mp4v).")

	processed = 0
	grabbed = 0
	try:
		while True:
			# Grab advances the stream cheaply; retrieve only when we actually
			# want to process/write the frame.
			if not cap.grab():
				break
			grabbed += 1

			# Prefer real timestamps when available (handles non-integer FPS better).
			t_s: float | None = None
			if use_pos_msec:
				pos_msec = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
				if pos_msec > 0:
					t_s = pos_msec / 1000.0
				else:
					# Some backends always report 0; fall back to FPS-based time.
					use_pos_msec = False

			if t_s is None:
				if in_fps > 0:
					t_s = (grabbed - 1) / float(in_fps)
				else:
					# No timing info at all; process every frame.
					t_s = next_sample_t_s

			if t_s + 1e-9 < next_sample_t_s:
				continue

			ret, frame_bgr = cap.retrieve()
			if not ret:
				continue

			# Move next sample time forward (catch up if we jumped ahead).
			while next_sample_t_s <= t_s + 1e-9:
				next_sample_t_s += sample_period_s

			masks = engine.infer_masks(frame_bgr)
			masks = _selective_masks(masks, cfg)

			# 4 separate overlays + titles
			vit_bgr = overlay_single_mask(frame_bgr, masks.vit_mask, color_bgr=(255, 0, 0), title="1) ViT-Hybrid")
			yolo_bgr = overlay_single_mask(frame_bgr, masks.yolo_mask, color_bgr=(0, 0, 255), title="2) YOLO")
			ufld_bgr = overlay_single_mask(frame_bgr, masks.ufld_mask, color_bgr=(0, 255, 0), title="3) UFLD")
			poly_bgr = overlay_single_mask(frame_bgr, masks.traditional_mask, alpha=0.35, color_bgr=(0, 255, 0), title="4) Polygon")

			# Output frame: 2x2 split (ViT, YOLO, UFLD, Polygon)
			split_bgr = make_split_2x2(frame_bgr, masks, with_titles=True)
			writer.write(split_bgr)
			processed += 1

			if processed % max(1, cfg.preview_every_n_frames) == 0:
				preview_slots[0].image(cv2.cvtColor(vit_bgr, cv2.COLOR_BGR2RGB), caption="1) ViT-Hybrid", use_container_width=True)
				preview_slots[1].image(cv2.cvtColor(yolo_bgr, cv2.COLOR_BGR2RGB), caption="2) YOLO", use_container_width=True)
				preview_slots[2].image(cv2.cvtColor(ufld_bgr, cv2.COLOR_BGR2RGB), caption="3) UFLD", use_container_width=True)
				preview_slots[3].image(cv2.cvtColor(poly_bgr, cv2.COLOR_BGR2RGB), caption="4) Polygon", use_container_width=True)

			if frame_count > 0:
				progress.progress(min(1.0, grabbed / frame_count))
				status.write(
					f"Processing… {grabbed}/{frame_count} frames (written: {processed}), output FPS: {out_fps:.2f}"
				)
	finally:
		cap.release()
		writer.release()

	status.write(f"Done. Wrote {processed} frames.")
	progress.progress(1.0)

	if cfg.export_to_disk and cfg.export_path is not None:
		cfg.export_path.parent.mkdir(parents=True, exist_ok=True)
		cfg.export_path.write_bytes(out_path.read_bytes())

	return out_path


def _page() -> None:
	st.set_page_config(page_title="LANESIGHT UI", layout="wide")
	st.title("LANESIGHT_UI (Streamlit)")
	st.caption("Thin UI layer: video IO + controls. All inference runs in LANESIGHT/video/infer_video.py")

	with st.sidebar:
		st.header("Modules")
		enable_vit = st.checkbox("Hybrid-ViT lane segmentation", value=True)
		enable_ufld = st.checkbox("UFLD lane detection", value=True)
		enable_traditional = st.checkbox("Traditional lane processing", value=False)
		enable_yolo = st.checkbox("YOLO vehicle detection", value=True)

		st.header("Output")
		out_fps = st.number_input("Output FPS", min_value=1.0, max_value=120.0, value=20.0, step=1.0)
		preview_every = st.number_input(
			"Preview every N written frames",
			min_value=1,
			max_value=200,
			value=10,
			step=1,
		)

		st.header("Export")
		export_to_disk = st.checkbox("Also export processed video to disk", value=False)
		export_path_str = st.text_input(
			"Export path (server-side)",
			value=str((REPO_ROOT / "LANESIGHT" / "test_output" / "processed.mp4").resolve()),
			disabled=not export_to_disk,
		)
		export_path = Path(export_path_str) if export_to_disk and export_path_str else None

		st.divider()
		run_clicked = st.button("Run", type="primary", use_container_width=True)

	col_left, col_right = st.columns([1, 1], gap="large")

	with col_left:
		st.subheader("Input")
		uploaded = st.file_uploader(
			"Upload dashcam video",
			type=["mp4", "avi", "mov", "mkv"],
			help="The UI only handles video IO; inference is in LANESIGHT.",
		)

	with col_right:
		st.subheader("Preview (Split 2x2)")
		r1c1, r1c2 = st.columns(2, gap="medium")
		r2c1, r2c2 = st.columns(2, gap="medium")
		preview_vit = r1c1.empty()
		preview_yolo = r1c2.empty()
		preview_ufld = r2c1.empty()
		preview_poly = r2c2.empty()

	st.subheader("Output Video")
	output_video_slot = st.empty()
	download_slot = st.empty()
	status = st.empty()
	progress = st.progress(0.0)

	if run_clicked:
		if uploaded is None:
			st.error("Please upload a video first.")
			return

		cfg = UiConfig(
			enable_vit=enable_vit,
			enable_ufld=enable_ufld,
			enable_traditional=enable_traditional,
			enable_yolo=enable_yolo,
			out_fps=float(out_fps),
			preview_every_n_frames=int(preview_every),
			export_to_disk=export_to_disk,
			export_path=export_path,
		)

		engine = get_engine()
		suffix = Path(uploaded.name).suffix or ".mp4"
		input_path = _write_uploaded_to_temp(uploaded.getvalue(), suffix=suffix)

		with st.spinner("Running LANESIGHT inference…"):
			out_path = process_video(
				input_path=input_path,
				cfg=cfg,
				engine=engine,
				progress=progress,
				status=status,
				preview_slots=(preview_vit, preview_yolo, preview_ufld, preview_poly),
			)

		video_bytes = out_path.read_bytes()
		output_video_slot.video(video_bytes)
		download_slot.download_button(
			"Download processed video",
			data=video_bytes,
			file_name="lanesight_processed.mp4",
			mime="video/mp4",
			use_container_width=True,
		)


if __name__ == "__main__":
	_page()

