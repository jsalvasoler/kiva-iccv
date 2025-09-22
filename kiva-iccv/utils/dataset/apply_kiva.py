import torch
from torchvision import transforms
from utils.dataset.transformations_kiva import (
    apply_counting,
    apply_reflection,
    apply_resizing,
    apply_rotation,
    paste_on_600,
)

canvas_resize = transforms.Resize((300, 300))


def _kiva_counting(
    img_A: torch.Tensor, img_C: torch.Tensor, true_param: str, incorrect_params: list[str]
) -> tuple:
    img_A_initial, img_B_correct, start_count, _ = apply_counting(img_A, true_param, type="train")

    img_C_initial, img_D_correct, start_count, _ = apply_counting(
        img_C, true_param, type="train", initial_count=start_count
    )

    _, img_E_incorrect, _, _ = apply_counting(
        img_C, incorrect_params[0], type="train", initial_count=start_count
    )
    _, img_F_incorrect, _, _ = apply_counting(
        img_C, incorrect_params[1], type="train", initial_count=start_count
    )
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )


def _kiva_reflect(
    img_A: torch.Tensor, img_C: torch.Tensor, true_param: str, incorrect_params: list[str]
) -> tuple:
    img_B_correct, _, _ = apply_reflection(img_A, true_param, type="train")
    img_D_correct, _, _ = apply_reflection(img_C, true_param, type="train")
    img_E_incorrect, _, _ = apply_reflection(img_C, incorrect_params[0], type="train")
    img_F_incorrect, _, _ = apply_reflection(img_C, incorrect_params[1], type="train")
    img_A_initial, img_C_initial = img_A, img_C
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )


def _kiva_resizing(
    img_A: torch.Tensor, img_C: torch.Tensor, true_param: str, incorrect_params: list[str]
) -> tuple:
    img_A, img_C = canvas_resize(img_A), canvas_resize(img_C)

    img_B_correct, _, _ = apply_resizing(img_A, true_param, type="train")
    img_D_correct, _, _ = apply_resizing(img_C, true_param, type="train")
    img_E_incorrect, _, _ = apply_resizing(img_C, incorrect_params[0], type="train")
    img_F_incorrect, _, _ = apply_resizing(img_C, incorrect_params[1], type="train")

    img_A_initial, img_C_initial = (
        paste_on_600(img_A),
        paste_on_600(img_C),
    )
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )


def _kiva_rotation(
    img_A: torch.Tensor, img_C: torch.Tensor, true_param: str, incorrect_params: list[str]
) -> tuple:
    img_B_correct, _, _ = apply_rotation(img_A, true_param, type="train")
    img_D_correct, _, _ = apply_rotation(img_C, true_param, type="train")
    img_E_incorrect, _, _ = apply_rotation(img_C, incorrect_params[0], type="train")
    img_F_incorrect, _, _ = apply_rotation(img_C, incorrect_params[1], type="train")
    img_A_initial, img_C_initial = img_A, img_C
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )
