FROM python:3.12-slim

WORKDIR /code

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY . .

ENV PYTHONPATH=/code/src

EXPOSE 5000

CMD [ "uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "5000" ]