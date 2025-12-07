#!/usr/bin/env bash
set -e

DB_HOST=${DB_HOST:-db}
DB_PORT=${DB_PORT:-5432}
TIMEOUT=${DB_WAIT_TIMEOUT:-60}

echo "Waiting for database at ${DB_HOST}:${DB_PORT} (timeout ${TIMEOUT}s)..."

for i in $(seq 1 ${TIMEOUT}); do
  if command -v pg_isready >/dev/null 2>&1; then
    pg_isready -h "${DB_HOST}" -p "${DB_PORT}" >/dev/null 2>&1 && break
  else
    # fallback: try TCP connect
    if (</dev/tcp/${DB_HOST}/${DB_PORT}) >/dev/null 2>&1; then
      break
    fi
  fi
  echo "  - still waiting (${i})"
  sleep 1
done

# final check
if command -v pg_isready >/dev/null 2>&1; then
  if ! pg_isready -h "${DB_HOST}" -p "${DB_PORT}" >/dev/null 2>&1; then
    echo "Timed out waiting for Postgres at ${DB_HOST}:${DB_PORT}" >&2
    exit 1
  fi
fi

echo "Database is up — starting the command"
exec "$@"
