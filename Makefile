SHELL := /bin/bash
TEACHER_MIX ?= 0.5

.PHONY: setup play arena train distill zip gate

setup:
	uv sync

play:
	uv run python -m harness.play --white . --black baselines/greedy $(if $(FEN),--fen "$(FEN)")

arena:
	uv run python -m harness.arena --opponent baselines/greedy --games 20

train:
	uv run python -m training.train --resume $(if $(TEACHER),--teacher-dataset "$(TEACHER)" --teacher-mix $(TEACHER_MIX))

distill:
	uv run python -m training.distill

zip:
	uv run python -m harness.package

gate:
	uv run ruff check .
	uv run mypy
	uv run python -m harness.arena --opponent baselines/random --games 2 --base-ms 5000
