import type { CohortBatch } from "./types";
import type { Upcall } from "./cohort";

/**
 * The default upcall: write a one-line summary of each consistent moment to the
 * console.
 *
 * THIS is the seam. Replace it with your own logic: apply `batch` to your store
 * and durably record `F` (ideally in one transaction); on restart, resume the
 * cohort `from` that `F`. Remember the contract (DESIGN.md, Idiom 3): returning
 * from here means only "delivered", never "durable" — durability is yours, and
 * delivery is at-least-once, so make your apply idempotent.
 */
export function consoleUpcall(label: string): Upcall {
  let lastHeartbeat = 0;
  return async (F: bigint, batch: CohortBatch): Promise<void> => {
    const changes = batch.reduce((n, v) => n + v.rows.length, 0);

    if (changes === 0) {
      // Idle advance (~1/s): the consistent moment still moved, and a real
      // consumer could record F here for a tighter resume. Heartbeat sparingly.
      const now = Date.now();
      if (now - lastHeartbeat >= 10_000) {
        console.log(`[${label}] consistent through F=${F} (idle)`);
        lastHeartbeat = now;
      }
      return;
    }

    const parts = batch.map((v) => {
      const rows = v.rows
        .map((r) => `${r.diff > 0n ? "+" : ""}${r.diff}×(${r.values.join(",")})`)
        .join(" ");
      return `${v.view}: ${rows}`;
    });
    console.log(`[${label}] consistent through F=${F} — ${parts.join(" | ")}`);
  };
}
