#!/bin/zsh
# Batched corpus sweep: ALL files in ONE sqllogictest invocation (one clusterd
# launch => one macOS firewall prompt), then the rebuild-failures in ONE main
# invocation (one more). ~2 prompts total instead of ~680. Per-file verdicts are
# recovered from the runner's `--- <file>` headers + indented PASS/FAIL summaries.
# NEVER rewrites checked-in slts.
set -u
WT=/Users/frankmcsherry/Projects/materialize/.claude/worktrees/optimizer-rebuild
SC=/Users/frankmcsherry/Projects/materialize/play/optimizer-rebuild/scorecard
OUT=$SC/corpus.md
cd $WT || exit 1
files=( test/sqllogictest/**/*.slt(.N) )
# Exclude recursion_limit.slt: it stack-overflows the dev (unoptimized) build
# on main AND rebuild (CI runs --optimized); in a shared process it aborts the
# whole run. Pre-existing, not rebuild-attributable; noted separately.
files=( ${files:#*recursion_limit.slt} )
N=${#files}

parse() {  # $1=logfile -> "PASS|FAIL|PANIC <file>" per file
  awk '
    /^--- / { f=$2; ord[++n]=f; fl[f]=0 }
    /^[[:space:]]+FAIL:/ { if (f!="") fl[f]=1 }
    /panicked/ { if (f!="") fl[f]=2 }
    END { for(i=1;i<=n;i++){ v=fl[ord[i]]; print (v==2?"PANIC":(v==1?"FAIL":"PASS")), ord[i] } }
  ' "$1"
}

# Pass 1: rebuild, all files, single invocation.
MZ_OPTIMIZER_REBUILD=logical bin/sqllogictest -- $files > /tmp/sweep_rb.log 2>&1
rb_exit=$?
parse /tmp/sweep_rb.log > /tmp/rb_verdicts.txt
headers=$(grep -c '^--- ' /tmp/sweep_rb.log)
grep -E '^(FAIL|PANIC) ' /tmp/rb_verdicts.txt | awk '{print $2}' > /tmp/rb_failed.txt
nfail=$(wc -l < /tmp/rb_failed.txt | tr -d ' ')

# Pass 2: main, only rebuild-failures, single invocation.
if [ "$nfail" -gt 0 ]; then
  bin/sqllogictest -- $(cat /tmp/rb_failed.txt) > /tmp/sweep_main.log 2>&1
  parse /tmp/sweep_main.log | grep -E '^(FAIL|PANIC) ' | awk '{print $2}' | sort > /tmp/main_failed.txt
else
  : > /tmp/main_failed.txt
fi

# Classify.
: > $OUT
print -- "# Batched corpus sweep (rebuild, main-controlled) — $(date '+%Y-%m-%d %H:%M')" >> $OUT
print -- "" >> $OUT
typeset -i PASS REGP=0 REGR=0 BOTH=0 PANIC=0
PASS=$(grep -c '^PASS ' /tmp/rb_verdicts.txt)
while read v f; do
  [ "$v" = PASS ] && continue
  if [ "$v" = PANIC ]; then PANIC+=1; print -- "PANIC       $f" >> $OUT; continue; fi
  if grep -qxF "$f" /tmp/main_failed.txt; then BOTH+=1; print -- "BOTHFAIL    $f" >> $OUT
  elif grep -qiE 'explain' "$f"; then REGP+=1; print -- "REGRESS_EXPLAIN $f" >> $OUT
  else REGR+=1; print -- "REGRESS_RESULT  $f" >> $OUT; fi
done < /tmp/rb_verdicts.txt
print -- "" >> $OUT
print -- "## headers_seen=$headers (expect $N), rb_exit=$rb_exit, rebuild_failures=$nfail" >> $OUT
print -- "## Totals over $N: pass=$PASS regress_with_explain=$REGP regress_result_NOEXPLAIN=$REGR bothfail=$BOTH panic=$PANIC" >> $OUT
print "batched done: N=$N headers=$headers pass=$PASS reg_explain=$REGP reg_result=$REGR bothfail=$BOTH panic=$PANIC rb_exit=$rb_exit"
