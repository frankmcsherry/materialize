#!/bin/zsh
# Full-corpus sweep, rebuild vs main control. macOS has no `timeout`, so we
# use a perl SIGALRM wrapper (exit 142 on timeout). Every rebuild FAIL is
# re-run on main so we report only rebuild-ATTRIBUTABLE deltas; files that
# fail on main too are pre-existing (flags/unsupported/flaky), not ours.
# Harness errors (build failure, command-not-found) are bucketed separately
# and abort the run if they hit the first few files. NEVER rewrites slts.
set -u
WT=/Users/frankmcsherry/Projects/materialize/.claude/worktrees/optimizer-rebuild
SC=/Users/frankmcsherry/Projects/materialize/play/optimizer-rebuild/scorecard
OUT=$SC/corpus.md
cd $WT || exit 1
: > $OUT
TO=300
runf() {  # $1=mode  $2=file
  local envv=""; [[ $1 == rebuild ]] && envv="MZ_OPTIMIZER_REBUILD=logical"
  perl -e 'alarm shift @ARGV; exec @ARGV' $TO env ${envv:+$envv} bin/sqllogictest -- "$2" >/tmp/cs.log 2>&1
  return $?
}
harness_err() { grep -qE "command not found|could not compile|^error\[E[0-9]|No such file or directory" /tmp/cs.log; }

typeset -i PASS=0 REGP=0 REGR=0 BOTH=0 PANIC=0 TO_=0 ERR=0 N=0
print -- "# Corpus sweep v2 (rebuild, main-controlled) — $(date '+%Y-%m-%d %H:%M')" >> $OUT
print -- "" >> $OUT
for f in test/sqllogictest/**/*.slt(.N); do
  N+=1
  runf rebuild "$f"; code=$?
  if harness_err; then
    ERR+=1; print -- "ERROR       $f :: $(tail -1 /tmp/cs.log)" >> $OUT
    if (( N <= 3 )); then print "ABORT: harness error on early file ($f):"; tail -4 /tmp/cs.log; exit 2; fi
    continue
  fi
  if (( code == 0 )); then PASS+=1; continue; fi
  if grep -q "panicked" /tmp/cs.log; then PANIC+=1; print -- "PANIC       $f" >> $OUT; continue; fi
  if (( code == 142 )); then TO_+=1; print -- "TIMEOUT     $f" >> $OUT; continue; fi
  hasx=no; grep -qiE "explain" "$f" && hasx=yes
  runf main "$f"; mcode=$?
  if harness_err; then ERR+=1; print -- "ERROR(main) $f" >> $OUT; continue; fi
  if (( mcode == 0 )); then
    if [[ $hasx == yes ]]; then REGP+=1; print -- "REGRESS_EXPLAIN $f" >> $OUT
    else REGR+=1; print -- "REGRESS_RESULT  $f" >> $OUT; fi
  else
    BOTH+=1; print -- "BOTHFAIL    $f" >> $OUT
  fi
done
print -- "" >> $OUT
print -- "## Totals over $N: pass=$PASS regress_with_explain=$REGP regress_result_NOEXPLAIN=$REGR bothfail_preexisting=$BOTH panic=$PANIC timeout=$TO_ harness_err=$ERR" >> $OUT
print "v2 done: N=$N pass=$PASS reg_explain=$REGP reg_result=$REGR bothfail=$BOTH panic=$PANIC timeout=$TO_ err=$ERR"
