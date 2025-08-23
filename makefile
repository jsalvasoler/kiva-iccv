fmt:
	uv tool run ruff format
	uv tool run ruff check --fix

test-train:
	uv run python kiva-iccv/train.py --do_train --do_test --train_on unit --validate_on unit --test_on unit

test-overfit-validation:
	uv run python kiva-iccv/train.py --do_train --do_test --train_on validation --validate_on validation_sample --test_on validation