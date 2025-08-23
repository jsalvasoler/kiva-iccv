fmt:
	uv tool run ruff format --check

test-train:
	uv run python kiva-iccv/train.py --do_train --do_test --train_on unit --validate_on unit --test_on unit