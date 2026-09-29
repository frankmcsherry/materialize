#!/bin/sh
# Bring the fixed-income board up from nothing: a Materialize container, the
# synthetic domain and board views, a Python venv, and board.py.
#
#   ./run.sh            then open http://localhost:8765/?trader=alice
#
# Env (defaults in brackets):
#   CONTAINER        Docker container name [mz-fi]. Reused if it already exists.
#   MZ_PORT          host port for SQL [16875]; system port is +2, HTTP is +3.
#                    Not 6875: other Materialize instances often live there.
#   BOARD_PORT       port for board.py [8765]
#   FI_CLUSTER_SIZE  replica size [200cc = 4 workers; see README "Replica size"]
#   FI_RETENTION, FI_PRICE_SLOTS   passed to load.sh (see its header)
#   IMAGE            [materialize/materialized:latest]
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
CONTAINER=${CONTAINER:-mz-fi}
MZ_PORT=${MZ_PORT:-16875}
SYS_PORT=$((MZ_PORT + 2))
HTTP_PORT=$((MZ_PORT + 3))
BOARD_PORT=${BOARD_PORT:-8765}
IMAGE=${IMAGE:-materialize/materialized:latest}
export FI_CLUSTER_SIZE=${FI_CLUSTER_SIZE:-200cc}
export PGOPTIONS="-c client_min_messages=warning"

listening() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "Reusing running container $CONTAINER."
elif docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "Starting existing container $CONTAINER."
    docker start "$CONTAINER" >/dev/null
else
    for p in $MZ_PORT $SYS_PORT $HTTP_PORT; do
        if listening "$p"; then
            echo "Port $p is already in use. Pick another base with MZ_PORT=..." >&2
            exit 1
        fi
    done
    echo "Starting $IMAGE as $CONTAINER on ports $MZ_PORT/$SYS_PORT/$HTTP_PORT."
    docker run -d --name "$CONTAINER" -p "$MZ_PORT:6875" -p "$SYS_PORT:6877" \
        -p "$HTTP_PORT:6878" "$IMAGE" >/dev/null
fi

PSQL="psql -h 127.0.0.1 -p $MZ_PORT -U materialize"
SYS="psql -h 127.0.0.1 -p $SYS_PORT -U mz_system -d materialize"
if ! command -v psql >/dev/null 2>&1; then
    PSQL="docker exec -i $CONTAINER psql -h localhost -p 6875 -U materialize"
    SYS="docker exec -i $CONTAINER psql -h localhost -p 6877 -U mz_system -d materialize"
fi
printf "Waiting for Materialize"
until $PSQL -X -q -c "SELECT 1" >/dev/null 2>&1; do printf "."; sleep 1; done
echo

PSQL="$PSQL" SYS="$SYS" PIVOTS=board "$HERE/load.sh"

VENV="$HERE/.venv"
if [ ! -x "$VENV/bin/python" ]; then
    echo "Creating $VENV."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" -q install 'psycopg[binary]'
fi

if listening "$BOARD_PORT"; then
    echo "Port $BOARD_PORT is in use (an old board.py?). Stop it or set BOARD_PORT." >&2
    exit 1
fi
printf "Waiting for the board views to hydrate"
until $PSQL -X -q -At -c "SELECT count(*) FROM materialize_demo.board_leaf" >/dev/null 2>&1; do printf "."; sleep 1; done
echo
MZ_DSN="host=127.0.0.1 port=$MZ_PORT user=materialize dbname=materialize" \
    exec "$VENV/bin/python" "$HERE/board.py" --port "$BOARD_PORT"
