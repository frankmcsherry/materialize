#!/bin/sh
# Stand up the fixed-income demo on a running Materialize.
#
#   PSQL  connection to the SQL port (default: localhost:6875 as materialize)
#   SYS   connection to the system port, for ALTER SYSTEM (localhost:6877)
#   FI_RETENTION    trade window, default '3 hours' (~88k positions)
#   FI_PRICE_SLOTS  each bond reprices every slots x 100ms, default 50
#   FI_CLUSTER_SIZE resize the quickstart cluster first, e.g. 200cc (4 workers).
#                   The Docker image defaults to 800cc (16 workers), which on
#                   this data is mostly coordination overhead: see results/workers.txt.
#   PIVOTS          which views to build: pivots (trader.py, default), board
#                   (board.py), or both. Both together cost more CPU.
#
# With the Docker image and no local psql:
#   PSQL="docker exec -i mz psql -h localhost -p 6875 -U materialize" \
#   SYS="docker exec -i mz psql -h localhost -p 6877 -U mz_system -d materialize" ./load.sh
set -e
PSQL=${PSQL:-"psql -h localhost -p 6875 -U materialize"}
SYS=${SYS:-"psql -h localhost -p 6877 -U mz_system -d materialize"}
HERE=$(cd "$(dirname "$0")" && pwd)
ASSETS=$HERE/../../assets

# The domain's clock ticks every 100ms. Without writes to tables, Materialize
# only advances time every default_timestamp_interval (1s by default).
$SYS -X -q -c "ALTER SYSTEM SET default_timestamp_interval = '100ms'"

if [ -n "$FI_CLUSTER_SIZE" ]; then
    $PSQL -X -q -c "ALTER CLUSTER quickstart SET (SIZE = '$FI_CLUSTER_SIZE')"
fi

# Everything lands in the `materialize_demo` schema, and every file is IF NOT
# EXISTS: re-running is safe, but changed knobs need assets/teardown.sql first.
# The domain reads the scaffold's `seconds`, which keep their rows for the
# scaffold's retention, so that must cover the trade window.
$PSQL -X -q -v ON_ERROR_STOP=1 -v retention="${FI_RETENTION:-3 hours}" < "$ASSETS/scaffold.sql"
$PSQL -X -q -v ON_ERROR_STOP=1 \
    -v fi_retention="${FI_RETENTION:-3 hours}" -v fi_price_slots="${FI_PRICE_SLOTS:-50}" \
    < "$ASSETS/domains/fixed_income.sql"
case "${PIVOTS:-pivots}" in
    pivots) $PSQL -X -q -v ON_ERROR_STOP=1 < "$HERE/pivots.sql"
            echo "Loaded. Try: python3 trader.py alice" ;;
    board)  $PSQL -X -q -v ON_ERROR_STOP=1 < "$HERE/board.sql"
            echo "Loaded. Try: python3 board.py, or ./run.sh to do everything" ;;
    both)   $PSQL -X -q -v ON_ERROR_STOP=1 < "$HERE/pivots.sql"
            $PSQL -X -q -v ON_ERROR_STOP=1 < "$HERE/board.sql"
            echo "Loaded. Try: python3 trader.py alice, or python3 board.py" ;;
    *)      echo "PIVOTS must be pivots, board or both" >&2; exit 1 ;;
esac
