"""Utilities to export the baseline ResNet-18 architecture as an image."""

import os
import sys
from typing import Optional

from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Optional: colorful view
try:
	import visualkeras
	from PIL import ImageFont
except ImportError:  # pragma: no cover - optional dependency
	visualkeras = None
	ImageFont = None


# Allow running as a script from project root
ROOT = os.getcwd()
sys.path.append(os.path.join(ROOT, "src"))

try:
	from .ResNet18 import build_resnet18
	from .ResidualBlock import ResidualBlock
except ImportError:
	from custom_regularizer.ResNet18 import build_resnet18
	from custom_regularizer.ResidualBlock import ResidualBlock


def export_colorful_baseline_architecture(
	output_path: Optional[str] = None,
) -> str:
	"""Generate a colorful layered view using visualkeras (if installed)."""
	if visualkeras is None:
		raise ImportError(
			"visualkeras (and Pillow) are required for colorful export. Install with: "
			"`poetry add visualkeras pillow`."
		)

	if output_path is None:
		results_dir = os.path.join(ROOT, "results")
		os.makedirs(results_dir, exist_ok=True)
		output_path = os.path.join(results_dir, "baseline_architecture_color.png")

	model = build_resnet18()

	# Some layers (e.g., InputLayer) may miss output_shape; add a best-effort field for rendering
	for layer in model.layers:
		if not hasattr(layer, "output_shape"):
			try:
				layer.output_shape = layer.compute_output_shape(layer.input_shape)
			except Exception:
				if hasattr(layer, "batch_input_shape"):
					layer.output_shape = layer.batch_input_shape
				elif hasattr(model, "input_shape"):
					layer.output_shape = model.input_shape

	# Optional font for labels
	font = None
	if ImageFont is not None:
		try:
			font = ImageFont.truetype("Arial.ttf", 16)
		except OSError:
			# Fallback to default PIL font if Arial is missing
			font = ImageFont.load_default()

	# Assign distinct colors per layer type for readability
	color_map = {
		keras.layers.Conv2D: {"fill": (79, 70, 229)},  # indigo
		keras.layers.BatchNormalization: {"fill": (16, 185, 129)},  # teal
		keras.layers.Activation: {"fill": (234, 179, 8)},  # amber
		keras.layers.Add: {"fill": (236, 72, 153)},  # pink
		keras.layers.GlobalAveragePooling2D: {"fill": (56, 189, 248)},  # sky
		keras.layers.Dense: {"fill": (248, 113, 113)},  # red
	}

	visualkeras.layered_view(
		model,
		to_file=output_path,
		legend=True,
		font=font,
		color_map=color_map,
		scale_xy=8,
		scale_z=1.5,  # exaggerate depth differences
		draw_volume=True,  # depth proportional to channel count
		max_z=512,
		spacing=30,
		background_fill=(245, 248, 255),
	)

	print(f"Saved colorful baseline architecture to {output_path}")
	return output_path


def _layer_channel_count(layer, model) -> int:
	"""Best-effort channel width detection for visualization."""
	if hasattr(layer, "filters") and layer.filters is not None:
		return int(layer.filters)
	if hasattr(layer, "output_shape") and layer.output_shape is not None:
		shape = layer.output_shape
		if isinstance(shape, tuple) and len(shape) > 2 and shape[-1] is not None:
			return int(shape[-1])
		if isinstance(shape, list) and shape and shape[0] is not None:
			last = shape[0]
			if isinstance(last, tuple) and len(last) > 2 and last[-1] is not None:
				return int(last[-1])
	if hasattr(model, "input_shape") and model.input_shape is not None:
		ishape = model.input_shape
		if isinstance(ishape, tuple) and len(ishape) > 2 and ishape[-1] is not None:
			return int(ishape[-1])
	return 1


def export_width_scaled_schematic(output_path: Optional[str] = None) -> str:
	"""Render a schematic where width scales with channel count using matplotlib."""
	if output_path is None:
		results_dir = os.path.join(ROOT, "results")
		os.makedirs(results_dir, exist_ok=True)
		output_path = os.path.join(results_dir, "baseline_architecture_schematic.png")

	model = build_resnet18()

	layers_info = []
	for layer in model.layers:
		layer_type = type(layer)
		channels = _layer_channel_count(layer, model)
		layers_info.append((layer.name, layer_type, channels))

	color_map = {
		keras.layers.InputLayer: "#fbbf24",  # amber
		keras.layers.Conv2D: "#4f46e5",  # indigo
		keras.layers.BatchNormalization: "#10b981",  # teal
		keras.layers.Activation: "#eab308",  # yellow
		keras.layers.Add: "#ec4899",  # pink
		keras.layers.GlobalAveragePooling2D: "#38bdf8",  # sky
		keras.layers.Dense: "#f87171",  # red
		ResidualBlock if "ResidualBlock" in globals() else object: "#9333ea",  # purple fallback
	}

	min_w, max_w = 20, 200
	max_channels = max(c for _, _, c in layers_info) if layers_info else 1

	widths = []
	colors = []
	labels = []
	positions = []
	current_x = 0
	for name, ltype, ch in layers_info:
		norm = ch / max_channels if max_channels else 0
		w = min_w + norm * (max_w - min_w)
		widths.append(w)
		positions.append(current_x)
		current_x += w + 8
		colors.append(color_map.get(ltype, "#6b7280"))
		labels.append(f"{name}\n{ch} ch")

	plt.figure(figsize=(6, 18))
	for pos, w, col, lab in zip(positions, widths, colors, labels):
		plt.bar(x=0, height=w, bottom=pos, width=0.8, color=col, edgecolor="black")
		font_size = max(6, min(12, w * 0.05))
		plt.text(
			0,
			pos + w / 2,
			lab,
			ha="center",
			va="center",
			fontsize=font_size,
			color="white",
			bbox={"facecolor": "black", "alpha": 0.35, "pad": 1, "edgecolor": "none"},
			clip_on=True,
		)

	plt.axis("off")
	plt.ylim(0, current_x)
	plt.xlim(-1.5, 1.5)
	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved width-scaled schematic to {output_path}")
	return output_path


def _add_box(ax, origin, size, color, edgecolor="black", alpha=1.0, label=None, zorder=1):
	"""Add a 3D box to the given axes."""
	ox, oy, oz = origin
	w, h, d = size
	verts = [
		[(ox, oy, oz), (ox + w, oy, oz), (ox + w, oy + h, oz), (ox, oy + h, oz)],  # bottom
		[(ox, oy, oz + d), (ox + w, oy, oz + d), (ox + w, oy + h, oz + d), (ox, oy + h, oz + d)],  # top
		[(ox, oy, oz), (ox + w, oy, oz), (ox + w, oy, oz + d), (ox, oy, oz + d)],  # front
		[(ox, oy + h, oz), (ox + w, oy + h, oz), (ox + w, oy + h, oz + d), (ox, oy + h, oz + d)],  # back
		[(ox, oy, oz), (ox, oy + h, oz), (ox, oy + h, oz + d), (ox, oy, oz + d)],  # left
		[(ox + w, oy, oz), (ox + w, oy + h, oz), (ox + w, oy + h, oz + d), (ox + w, oy, oz + d)],  # right
	]
	box = Poly3DCollection(verts, facecolors=color, edgecolors=edgecolor, linewidths=0.6, alpha=alpha, zorder=zorder)
	ax.add_collection3d(box)
	if label:
		xc = ox + w / 2
		yc = oy + h / 2
		zc = oz + d / 2
		ax.text(xc, yc, zc, label, ha="center", va="center", fontsize=7, color="white", zorder=zorder + 1)


def export_detailed_3d(output_path: Optional[str] = None) -> str:
	"""3D view with per-block internals (Conv/BN) shown inside residual blocks."""
	if output_path is None:
		results_dir = os.path.join(ROOT, "results")
		os.makedirs(results_dir, exist_ok=True)
		output_path = os.path.join(results_dir, "baseline_architecture_3d.png")

	model = build_resnet18()
	blocks = []
	for layer in model.layers:
		if isinstance(layer, ResidualBlock):
			blocks.append((layer.name or "res_block", "res", layer.filters))
		elif isinstance(layer, keras.layers.Conv2D):
			blocks.append((layer.name, "conv", layer.filters))
		elif isinstance(layer, keras.layers.BatchNormalization):
			blocks.append((layer.name, "bn", _layer_channel_count(layer, model)))
		elif isinstance(layer, keras.layers.Activation):
			blocks.append((layer.name, "act", _layer_channel_count(layer, model)))
		elif isinstance(layer, keras.layers.GlobalAveragePooling2D):
			blocks.append((layer.name, "gap", _layer_channel_count(layer, model)))
		elif isinstance(layer, keras.layers.Dense):
			blocks.append((layer.name, "dense", _layer_channel_count(layer, model)))
		elif isinstance(layer, keras.layers.InputLayer):
			blocks.append((layer.name, "input", _layer_channel_count(layer, model)))

	# Scaling settings
	min_w, max_w = 0.6, 2.2
	min_h, max_h = 0.9, 2.2
	min_d, max_d = 0.5, 1.6
	max_channels = max(c for _, _, c in blocks) if blocks else 1
	colors = {
		"input": "#fbbf24",
		"conv": "#4f46e5",
		"bn": "#10b981",
		"act": "#eab308",
		"res": "#9333ea",
		"gap": "#38bdf8",
		"dense": "#f87171",
	}

	fig = plt.figure(figsize=(12, 6))
	ax = fig.add_subplot(111, projection="3d")

	x_cursor = 0.0
	input_anchor = None
	for name, kind, ch in blocks:
		norm = ch / max_channels if max_channels else 0
		w = min_w + norm * (max_w - min_w)
		h = min_h + norm * (max_h - min_h)
		d = min_d + norm * (max_d - min_d)
		if kind == "input":
			w, h, d = max(w, 1.4), max(h, 1.2), max(d, 1.0)
		color = colors.get(kind, "#6b7280")

		if kind == "res":
			# Residual block: render outer shell + internal conv/bn/act slabs
			_outer_w, _outer_h, _outer_d = w * 1.05, h * 1.05, max(d * 1.35, 0.6)
			_add_box(ax, (x_cursor, 0, 0), (_outer_w, _outer_h, _outer_d), color, alpha=0.9, label=name)

			inner_gap = 0.05
			usable_d = _outer_d - 7 * inner_gap
			conv_d = usable_d * 0.32
			bn_d = usable_d * 0.08
			act_d = usable_d * 0.08

			z_pos = inner_gap
			slabs = [
				("conv3x3", colors.get("conv"), conv_d),
				("bn", colors.get("bn"), bn_d),
				("relu", colors.get("act"), act_d),
				("conv3x3", colors.get("conv"), conv_d),
				("bn", colors.get("bn"), bn_d),
				("relu", colors.get("act"), act_d),
			]
			for label, col, depth in slabs:
				_add_box(
					ax,
					(x_cursor + 0.05, 0.05, z_pos),
					(w * 0.9, h * 0.9, depth),
					col,
					alpha=0.95,
					label=label,
				)
				z_pos += depth + inner_gap
		else:
			_add_box(ax, (x_cursor, 0, 0), (w, h, d), color, alpha=0.95, label=name)

		if kind == "input" and input_anchor is None:
			input_anchor = (x_cursor, w, h, d)

		x_cursor += w + 0.8

	ax.set_axis_off()
	ax.view_init(elev=12, azim=110)
	max_range = max(x_cursor, 8)
	ax.set_xlim(0, max_range)
	ax.set_ylim(-2, 4)
	ax.set_zlim(-0.5, 3.5)
	ax.set_box_aspect((max_range, 3, 2))

	# Feed direction arrow starting near input block
	if input_anchor:
		in_x, in_w, in_h, in_d = input_anchor
		start_x = max(in_x - in_w * 0.6, -0.5)  # anchor to the left of input block
		start_y = in_h * 0.8                    # above center
		start_z = in_d * 0.6                    # mid-depth
	else:
		start_x, start_y, start_z = -0.4, 0.5, 0.25

	arrow_len = max_range * 0.8
	ax.quiver(start_x, start_y, start_z, arrow_len, 0, 0, color="black", arrow_length_ratio=0.06, linewidth=2)
	ax.text(start_x + arrow_len * 0.5, start_y + 0.12, start_z + 0.05, "forward pass", ha="center", va="center", fontsize=9, color="black")

	# 2D legend for colors
	legend_patches = [
		Patch(color=colors[k], label=k) for k in ["input", "conv", "bn", "act", "res", "gap", "dense"]
	]
	ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(0, 1))

	plt.tight_layout()
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved 3D detailed view to {output_path}")
	return output_path


def export_single_block_sample(output_path: Optional[str] = None) -> str:
	"""Render a close-up 3D view of the largest residual block with internals."""
	if output_path is None:
		results_dir = os.path.join(ROOT, "results")
		os.makedirs(results_dir, exist_ok=True)
		output_path = os.path.join(results_dir, "baseline_architecture_block_sample.png")

	model = build_resnet18()
	res_layers = [l for l in model.layers if isinstance(l, ResidualBlock)]
	if not res_layers:
		print("No residual blocks found; skipping block sample.")
		return output_path

	target = max(res_layers, key=lambda l: getattr(l, "filters", 0))
	ch = getattr(target, "filters", 1)

	# Dimensions for internals only (no shell)
	w = 2.0
	h = 2.0
	d = 1.6
	colors = {
		"conv": "#4f46e5",
		"bn": "#10b981",
		"act": "#eab308",
	}

	fig = plt.figure(figsize=(6, 6))
	ax = fig.add_subplot(111, projection="3d")

	inner_gap = 0.05
	usable_d = d - 5 * inner_gap
	conv_d = usable_d * 0.36
	bn_d = usable_d * 0.08
	act_d = usable_d * 0.08
	z_pos = inner_gap
	slabs = [
		("conv3x3", colors["conv"], conv_d),
		("bn", colors["bn"], bn_d),
		("relu", colors["act"], act_d),
		("conv3x3", colors["conv"], conv_d),
		("bn", colors["bn"], bn_d),
		("relu", colors["act"], act_d),
	]
	for label, col, depth in slabs:
		_add_box(
			ax,
			(0, 0, z_pos),
			(w, h, depth),
			col,
			alpha=0.95,
			label=label,
		)
		z_pos += depth + inner_gap

	# Feed direction arrow (front to back along z), anchored from left side, above center
	arrow_len = z_pos + 0.2
	start_x, start_y, start_z = -0.4, h * 0.75, 0.05
	ax.quiver(start_x, start_y, start_z, 0, 0, arrow_len, color="black", arrow_length_ratio=0.08, linewidth=2)
	ax.text(start_x, start_y + 0.05, start_z + arrow_len * 0.6, "forward", ha="center", va="center", fontsize=9, color="black")

	ax.set_axis_off()
	ax.view_init(elev=18, azim=105)
	ax.set_xlim(-0.2, w * 1.05)
	ax.set_ylim(-0.2, h * 1.2)
	ax.set_zlim(-0.2, z_pos + 0.4)
	ax.set_box_aspect((w * 1.05, h * 1.2, z_pos + 0.4))

	legend_patches = [
		Patch(color=colors["conv"], label="Conv3x3"),
		Patch(color=colors["bn"], label="BatchNorm"),
		Patch(color=colors["act"], label="ReLU"),
	]
	ax.legend(handles=legend_patches, loc="upper left", bbox_to_anchor=(0, 1))

	plt.tight_layout()
	plt.savefig(output_path, dpi=400, bbox_inches="tight")
	plt.close()
	print(f"Saved block sample view to {output_path}")
	return output_path


def export_baseline_architecture(
	output_path: Optional[str] = None, colorful: bool = False
) -> str:
	"""Build the baseline model and save its architecture image.

	If `colorful=True` and visualkeras is available, a richer layered view is used.
	Otherwise, falls back to keras.utils.plot_model.
	"""

	if colorful and visualkeras is not None:
		return export_colorful_baseline_architecture(output_path)

	if output_path is None:
		results_dir = os.path.join(ROOT, "results")
		os.makedirs(results_dir, exist_ok=True)
		output_path = os.path.join(results_dir, "baseline_architecture.png")

	model = build_resnet18()

	keras.utils.plot_model(
		model,
		to_file=output_path,
		show_shapes=True,
		show_layer_names=True,
		expand_nested=True,
		dpi=300,
		rankdir="TB",
	)

	print(f"Saved baseline architecture to {output_path}")
	return output_path


def main():
	try:
		export_baseline_architecture(colorful=True)
	except ImportError as exc:
		print(f"Colorful export unavailable: {exc}. Falling back to plot_model.")
		export_baseline_architecture(colorful=False)

	# Always emit a width-scaled schematic for clearer channel differences
	export_width_scaled_schematic()

	# 3D detailed view with per-block internals
	export_detailed_3d()

	# Single largest residual block close-up
	export_single_block_sample()


if __name__ == "__main__":
	main()
