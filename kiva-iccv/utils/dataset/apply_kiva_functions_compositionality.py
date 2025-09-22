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


def _kiva_functions_compositionality_counting_reflect(
    img_A_base: torch.Tensor,
    img_C_base: torch.Tensor,
    true_count_param: str,
    true_reflect_param: str,
    incorrect_reflect_param: str,
    incorrect_count_param: str,
    start_count_A: int,
    start_count_C: int,
    start_reflect_A: str,
    start_reflect_C: str,
) -> tuple:
    def make_initial(image: torch.Tensor, start_count: int, start_reflect: str) -> torch.Tensor:
        reflected, _, _, _ = apply_reflection(image, start_reflect, type="train")
        initial_grid, _, _, _ = apply_counting(
            reflected, "+1", type="train", initial_count=start_count
        )
        return initial_grid, reflected

    def apply_true_chain(
        image: torch.Tensor, start_count: int, true_count: str, true_reflect: str
    ) -> torch.Tensor:
        reflected_correct, _, _, _ = apply_reflection(image, true_reflect, type="train")
        _, out, _, _ = apply_counting(
            reflected_correct, true_count, type="train", initial_count=start_count
        )
        return out

    img_A_initial, img_A_reflected = make_initial(img_A_base, start_count_A, start_reflect_A)
    img_C_initial, img_C_reflected = make_initial(img_C_base, start_count_C, start_reflect_C)

    img_B_correct = apply_true_chain(
        img_A_reflected, start_count_A, true_count_param, true_reflect_param
    )
    img_D_correct = apply_true_chain(
        img_C_reflected, start_count_C, true_count_param, true_reflect_param
    )

    img_E_incorrect = apply_true_chain(
        img_C_reflected, start_count_C, true_count_param, incorrect_reflect_param
    )

    img_F_incorrect = apply_true_chain(
        img_C_reflected, start_count_C, incorrect_count_param, true_reflect_param
    )
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )


def _kiva_functions_compositionality_counting_resizing(
    img_A_base: torch.Tensor,
    img_C_base: torch.Tensor,
    true_count_param: str,
    true_resizing_param: str,
    incorrect_resizing_param: str,
    incorrect_count_param: str,
    start_count_A: int,
    start_count_C: int,
    start_resizing_A: str,
    start_resizing_C: str,
) -> tuple:
    def make_initial(image: torch.Tensor, start_count: int, start_resize: str) -> torch.Tensor:
        image = canvas_resize(image)
        img_base_resized, _, _ = apply_resizing(image, start_resize, type="train")
        tmp_600 = paste_on_600(img_base_resized)
        img_out, _, _, _ = apply_counting(tmp_600, "+1", type="train", initial_count=start_count)
        return img_out, img_base_resized

    def apply_true_chain(
        image: torch.Tensor, start_count: int, true_count: str, true_resize: str
    ) -> torch.Tensor:
        img_temp, _, _ = apply_resizing(image, true_resize, type="train")
        img_temp = paste_on_600(img_temp)
        _, img_out, _, _ = apply_counting(
            img_temp, true_count, type="train", initial_count=start_count
        )
        return img_out

    img_A_initial, img_A_base_resized = make_initial(img_A_base, start_count_A, start_resizing_A)
    img_C_initial, img_C_base_resized = make_initial(img_C_base, start_count_C, start_resizing_C)

    img_B_correct = apply_true_chain(
        img_A_base_resized, start_count_A, true_count_param, true_resizing_param
    )
    img_D_correct = apply_true_chain(
        img_C_base_resized, start_count_C, true_count_param, true_resizing_param
    )

    img_E_incorrect = apply_true_chain(
        img_C_base_resized, start_count_C, true_count_param, incorrect_resizing_param
    )
    img_F_incorrect = apply_true_chain(
        img_C_base_resized, start_count_C, incorrect_count_param, true_resizing_param
    )

    img_A_initial, img_C_initial = paste_on_600(img_A_initial), paste_on_600(img_C_initial)
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )


def _kiva_functions_compositionality_counting_rotation(
    img_A_base: torch.Tensor,
    img_C_base: torch.Tensor,
    true_count_param: str,
    true_rotation_param: str,
    incorrect_rotation_param: str,
    incorrect_count_param: str,
    start_count_A: int,
    start_count_C: int,
    start_rotation_A: str,
    start_rotation_C: str,
) -> tuple:
    def make_initial(image: torch.Tensor, start_count: int, start_rotation: str) -> torch.Tensor:
        _, rotated, _, _ = apply_rotation(
            image, start_rotation, type="train", initial_rotation="+0"
        )
        img_out, _, _, _ = apply_counting(rotated, "+1", type="train", initial_count=start_count)
        return img_out, rotated

    def apply_true_chain(
        image: torch.Tensor, start_count: int, true_count: str, true_rotation: str
    ) -> torch.Tensor:
        _, img_temp, _, _ = apply_rotation(
            image, true_rotation, type="train", initial_rotation="+0"
        )
        _, img_out, _, _ = apply_counting(
            img_temp, true_count, type="train", initial_count=start_count
        )
        return img_out

    img_A_initial, img_A_rotated = make_initial(img_A_base, start_count_A, start_rotation_A)
    img_C_initial, img_C_rotated = make_initial(img_C_base, start_count_C, start_rotation_C)

    img_B_correct = apply_true_chain(
        img_A_rotated, start_count_A, true_count_param, true_rotation_param
    )
    img_D_correct = apply_true_chain(
        img_C_rotated, start_count_C, true_count_param, true_rotation_param
    )
    img_E_incorrect = apply_true_chain(
        img_C_rotated, start_count_C, true_count_param, incorrect_rotation_param
    )
    img_F_incorrect = apply_true_chain(
        img_C_rotated, start_count_C, incorrect_count_param, true_rotation_param
    )
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )


def _kiva_functions_compositionality_reflect_resizing(
    img_A_base: torch.Tensor,
    img_C_base: torch.Tensor,
    true_reflect_param: str,
    true_resizing_param: str,
    incorrect_reflect_param: str,
    incorrect_resizing_param: str,
    start_reflect_A: str,
    start_reflect_C: str,
    start_resizing_A: str,
    start_resizing_C: str,
) -> tuple:
    def apply_reflection_and_resizing(
        image: torch.Tensor, reflect_param: str, resizing_param: str
    ) -> torch.Tensor:
        img_temp, _, _, _ = apply_reflection(image, reflect_param, type="train")
        img_out, _, _ = apply_resizing(img_temp, resizing_param, type="train")
        return img_out

    img_A_base, img_C_base = (
        canvas_resize(img_A_base),
        canvas_resize(img_C_base),
    )

    img_A_initial = apply_reflection_and_resizing(img_A_base, start_reflect_A, start_resizing_A)
    img_C_initial = apply_reflection_and_resizing(img_C_base, start_reflect_C, start_resizing_C)

    img_B_correct = apply_reflection_and_resizing(
        img_A_initial, true_reflect_param, true_resizing_param
    )
    img_D_correct = apply_reflection_and_resizing(
        img_C_initial, true_reflect_param, true_resizing_param
    )
    img_E_incorrect = apply_reflection_and_resizing(
        img_C_initial, true_reflect_param, incorrect_resizing_param
    )
    img_F_incorrect = apply_reflection_and_resizing(
        img_C_initial, incorrect_reflect_param, true_resizing_param
    )

    img_A_initial, img_C_initial = paste_on_600(img_A_initial), paste_on_600(img_C_initial)
    img_B_correct, img_D_correct, img_E_incorrect, img_F_incorrect = (
        paste_on_600(img_B_correct),
        paste_on_600(img_D_correct),
        paste_on_600(img_E_incorrect),
        paste_on_600(img_F_incorrect),
    )
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )


def _kiva_functions_compositionality_resizing_rotation(
    img_A_base: torch.Tensor,
    img_C_base: torch.Tensor,
    true_resizing_param: str,
    true_rotation_param: str,
    incorrect_resizing_param: str,
    incorrect_rotation_param: str,
    start_resizing_A: str,
    start_resizing_C: str,
    start_rotation_A: str,
    start_rotation_C: str,
) -> tuple:
    def apply_resizing_and_rotation(
        image: torch.Tensor, resizing_param: str, rotation_param: str
    ) -> torch.Tensor:
        _, img_temp, _, _ = apply_rotation(
            image, rotation_param, type="train", initial_rotation="+0"
        )
        img_temp, _, _ = apply_resizing(img_temp, resizing_param, type="train")
        return img_temp

    img_A_base, img_C_base = (
        canvas_resize(img_A_base),
        canvas_resize(img_C_base),
    )

    img_A_initial = apply_resizing_and_rotation(img_A_base, start_resizing_A, start_rotation_A)
    img_C_initial = apply_resizing_and_rotation(img_C_base, start_resizing_C, start_rotation_C)

    img_B_correct = apply_resizing_and_rotation(
        img_A_initial, true_resizing_param, true_rotation_param
    )
    img_D_correct = apply_resizing_and_rotation(
        img_C_initial, true_resizing_param, true_rotation_param
    )
    img_E_incorrect = apply_resizing_and_rotation(
        img_C_initial, true_resizing_param, incorrect_rotation_param
    )
    img_F_incorrect = apply_resizing_and_rotation(
        img_C_initial, incorrect_resizing_param, true_rotation_param
    )

    # paste all on 600
    img_A_initial, img_C_initial = paste_on_600(img_A_initial), paste_on_600(img_C_initial)
    img_B_correct, img_D_correct, img_E_incorrect, img_F_incorrect = (
        paste_on_600(img_B_correct),
        paste_on_600(img_D_correct),
        paste_on_600(img_E_incorrect),
        paste_on_600(img_F_incorrect),
    )
    return (
        img_A_initial,
        img_B_correct,
        img_C_initial,
        img_D_correct,
        img_E_incorrect,
        img_F_incorrect,
    )
