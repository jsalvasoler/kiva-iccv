fmt:
	uv tool run ruff format
	uv tool run ruff check --fix

test-train-unit:
	uv run --env-file .env python kiva-iccv/train.py --do_train --do_test --train_on unit --validate_on unit --test_on unit

test-overfit-validation:
	uv run --env-file .env python kiva-iccv/train.py --do_train --do_test --train_on validation --test_on validation

train-generic:
	uv run --env-file .env python kiva-iccv/train.py --do_train --do_test --test_on validation --use_neptune --epochs 50 --batch_size 128 --freeze_encoder