import { loadConfig } from "./config";
import { Cohort } from "./cohort";
import { consoleUpcall } from "./console-sink";

/**
 * Demo wiring. Builds one cohort over the configured views and prints a line per
 * consistent moment. Pass `--from <F>` to resume instead of starting fresh.
 *
 * In a real consumer the `from` frontier comes back out of YOUR store (committed
 * in the same transaction as the data); here we just take it on the command
 * line so you can see resume behavior:
 *
 *   npm run dev                       # fresh: snapshot every view, then stream
 *   npm run dev -- config.json --from 1700000000000   # resume from a recorded F
 */
async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const configPath = args.find((a) => !a.startsWith("--")) ?? "config.json";
  const fromIdx = args.indexOf("--from");
  const from = fromIdx >= 0 ? BigInt(args[fromIdx + 1]) : null;

  const cfg = loadConfig(configPath);
  const upcall = consoleUpcall("demo");
  const opts = { fetchBatch: cfg.fetchBatch };

  const cohort =
    from === null
      ? await Cohort.fresh(cfg.mzConn, cfg.views, upcall, opts)
      : await Cohort.resuming(cfg.mzConn, cfg.views, from, upcall, opts);

  console.log(
    from === null
      ? `[demo] fresh start over ${cfg.views.join(", ")}`
      : `[demo] resume from F=${from} over ${cfg.views.join(", ")}`,
  );
  console.log("[demo] streaming consistent moments (Ctrl-C to exit)");

  // The FETCH loops keep the process alive; `failed` only settles by rejecting.
  // If a stream errors (e.g. AS OF below the RETAIN HISTORY window) we crash
  // loudly rather than act on a partial moment — restart resumes from your
  // durable F. There is no graceful stop in this recipe: Ctrl-C exits.
  await cohort.failed;
}

main().catch((err) => {
  console.error("[demo] fatal:", err);
  process.exit(1);
});
