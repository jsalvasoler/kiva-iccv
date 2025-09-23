import torch
from torchvision import transforms
from utils.dataset.transformations_kiva_adults import (
    apply_counting,
    apply_reflection,
    apply_resizing,
    apply_rotation,
    paste_on_600,
)

canvas_resize = transforms.Resize((300, 300))


def _kiva_functions_counting(
    img_A: torch.Tensor,
    img_C: torch.Tensor,
    true_param: str,
    incorrect_params: list[str],
    start_transformations: list[str],
) -> tuple:
    img_A_initial, img_B_correct, _, _ = apply_counting(
        img_A, true_param, type="train", initial_count=start_transformations[0]
    )
    img_C_initial, img_D_correct, _, _ = apply_counting(
        img_C, true_param, type="train", initial_count=start_transformations[1]
    )

    _, img_E_incorrect, _, _ = apply_counting(
        img_C, incorrect_params[0], type="train", initial_count=start_transformations[1]
    )
    _, img_F_incorrect, _, _ = apply_counting(
        img_C, incorrect_params[1], type="train", initial_count=start_transformations[1]
    )
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )


def _kiva_functions_reflect(
    img_A: torch.Tensor,
    img_C: torch.Tensor,
    true_param: str,
    incorrect_params: list[str],
    start_transformations: list[str],
) -> tuple:
    img_A_initial, _, _, _ = apply_reflection(img_A, start_transformations[0], type="train")
    img_B_correct, _, _, _ = apply_reflection(img_A_initial, true_param, type="train")

    img_C_initial, _, _, _ = apply_reflection(img_C, start_transformations[1], type="train")
    img_D_correct, _, _, _ = apply_reflection(img_C_initial, true_param, type="train")
    img_E_incorrect, _, _, _ = apply_reflection(img_C_initial, incorrect_params[0], type="train")
    img_F_incorrect, _, _, _ = apply_reflection(img_C_initial, incorrect_params[1], type="train")
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )


def _kiva_functions_resizing(
    img_A: torch.Tensor,
    img_C: torch.Tensor,
    true_param: str,
    incorrect_params: list[str],
    start_transformations: list[str],
) -> tuple:
    img_A, img_C = canvas_resize(img_A), canvas_resize(img_C)
    img_A_initial, _, _ = apply_resizing(img_A, start_transformations[0], type="train")
    img_C_initial, _, _ = apply_resizing(img_C, start_transformations[1], type="train")

    def apply_resizing_and_paste(image: torch.Tensor, param: str) -> torch.Tensor:
        img_out, _, _ = apply_resizing(image, param, type="train")
        return paste_on_600(img_out)

    img_B_correct = apply_resizing_and_paste(img_A_initial, true_param)
    img_D_correct = apply_resizing_and_paste(img_C_initial, true_param)
    img_E_incorrect = apply_resizing_and_paste(img_C_initial, incorrect_params[0])
    img_F_incorrect = apply_resizing_and_paste(img_C_initial, incorrect_params[1])

    # paste on 600 everything
    img_A_initial, img_C_initial = paste_on_600(img_A_initial), paste_on_600(img_C_initial)
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )


def _kiva_functions_rotation(
    img_A: torch.Tensor,
    img_C: torch.Tensor,
    true_param: str,
    incorrect_params: list[str],
    start_transformations: list[str],
) -> tuple:
    _, img_A_initial, _, _ = apply_rotation(
        img_A, start_transformations[0], type="train", initial_rotation="+0"
    )
    _, img_B_correct, _, _ = apply_rotation(
        img_A_initial, true_param, type="train", initial_rotation="+0"
    )
    _, img_C_initial, _, _ = apply_rotation(
        img_C, start_transformations[1], type="train", initial_rotation="+0"
    )
    _, img_D_correct, _, _ = apply_rotation(
        img_C_initial, true_param, type="train", initial_rotation="+0"
    )
    _, img_E_incorrect, _, _ = apply_rotation(
        img_C_initial, incorrect_params[0], type="train", initial_rotation="+0"
    )
    _, img_F_incorrect, _, _ = apply_rotation(
        img_C_initial, incorrect_params[1], type="train", initial_rotation="+0"
    )
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )
