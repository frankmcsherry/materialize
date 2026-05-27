# Getting a live Materialize to demo against

The bridge needs two databases: a **Materialize** (source, pgwire on `:6875`)
and a **Postgres** (downstream sink). Pick one of the following ways to get a
Materialize.

## Option A — Materialize emulator (recommended for demos)

The emulator is a single Docker image that bundles all of Materialize. It needs
a working Docker daemon. The compose file also starts the downstream Postgres,
so this is the whole stack in one command:

```bash
docker compose -f harness/docker-compose.yml up -d
# wait for healthchecks, then:
psql postgres://materialize@localhost:6875/materialize -f harness/setup.sql
```

Use this `config.json` (matches the compose ports):

```json
{
  "mzConn": "postgres://materialize@localhost:6875/materialize",
  "pgConn": "postgres://postgres:postgres@localhost:5432/postgres",
  "bridgeId": "demo",
  "retainHistory": "1h",
  "fetchBatch": 1000,
  "views": [
    { "view": "mv_accounts", "target": "mv_accounts" },
    { "view": "mv_names", "target": "mv_names" },
    { "view": "mv_balances", "target": "mv_balances" }
  ]
}
```

Then `npm install && npm run dev`, and drive changes with
`psql ... -f harness/changes.sql`.

### Note on nested Docker (OrbStack Linux VMs)

Running the emulator *inside* an OrbStack Linux VM currently fails at container
start with:

```
runc create failed: ... bpf_prog_query(BPF_CGROUP_DEVICE) failed: operation not permitted
```

The VM kernel blocks the cgroup-device `bpf()` program that runc installs; this
is independent of storage driver, cgroup driver, or `--privileged`. **Run the
emulator from a session that has a real Docker daemon** — e.g. the OrbStack host
itself, or any normal machine — and point the bridge at it (the ports are
published on localhost). This is the "extra-VM session pilots it" path.

## Option B — build environmentd from source (works inside the VM)

This avoids Docker entirely and uses a plain Postgres as Materialize's metadata
store, so it runs fully inside the VM.

```bash
# 1. a Postgres for Materialize's own metadata (separate from the sink!)
#    e.g. a throwaway local cluster on :55432 with a 'materialize' db and
#    'consensus'/'tsoracle' schemas.
export MZDEV_POSTGRES='postgres://postgres@127.0.0.1:55432/materialize'

# 2. build + run (debug build; first compile is slow)
bin/environmentd --reset

# 3. Materialize SQL is on :6875; point the bridge's mzConn there and its
#    pgConn at your downstream Postgres.
```
