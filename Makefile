all: run

install:
	uv sync

run:
	uv run python -m src

# debug:

clean:
	uv cache clean

# lint: