from __future__ import annotations

from typing import Any


MODEL_INPUT_TENSOR_CHOICES = ("image", "raw")


def coerce_model_input_tensor(value: Any, *, default: str | None = None) -> str:
    if value is None:
        if default is None:
            raise ValueError("model_input_tensor is required")
        value = default
    value = str(value)
    if value == "rgb":
        value = "image"
    if value not in MODEL_INPUT_TENSOR_CHOICES:
        raise ValueError(
            f"Unsupported model_input_tensor={value!r}; "
            f"expected one of {', '.join(MODEL_INPUT_TENSOR_CHOICES)}"
        )
    return value


def model_input_tensor_from_resolved(resolved_or_tensor: Any) -> str:
    if hasattr(resolved_or_tensor, "model_input_tensor"):
        return coerce_model_input_tensor(getattr(resolved_or_tensor, "model_input_tensor"))
    return coerce_model_input_tensor(resolved_or_tensor)


def _sample_keys(sample: dict[str, Any]) -> str:
    return ", ".join(sorted(str(key) for key in sample.keys()))


def select_model_input(
    sample: dict[str, Any],
    resolved_or_tensor: Any,
    *,
    dataset_family: str | None = None,
    sample_source: str | None = None,
    add_batch_dim: bool = False,
):
    model_input_tensor = model_input_tensor_from_resolved(resolved_or_tensor)
    if model_input_tensor not in sample or sample[model_input_tensor] is None:
        context = []
        if dataset_family:
            context.append(f"dataset_family={dataset_family}")
        if sample_source:
            context.append(f"sample_source={sample_source}")
        context.append(f"model_input_tensor={model_input_tensor}")
        context.append(f"available_keys=[{_sample_keys(sample)}]")
        raise KeyError("Sample is missing the configured model input tensor: " + " ".join(context))

    tensor = sample[model_input_tensor]
    if add_batch_dim and getattr(tensor, "ndim", None) == 3:
        tensor = tensor.unsqueeze(0)
    return tensor
