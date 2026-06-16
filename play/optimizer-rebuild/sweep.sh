#!/bin/zsh
# Scorecard sweep: run slt files under MZ_OPTIMIZER_REBUILD=logical, classify
# divergences. NEVER rewrites checked-in slt files. Usage: sweep.sh <name> <files...>
NAME=$1; shift
WT=/Users/frankmcsherry/Projects/materialize/.claude/worktrees/optimizer-rebuild
OUT=/Users/frankmcsherry/Projects/materialize/play/optimizer-rebuild/scorecard/$NAME.md
cd $WT
echo "# Sweep: $NAME ($(date '+%Y-%m-%d %H:%M'))" > $OUT
echo "" >> $OUT
PASS=0; PLAN=0; RESULT=0; ERR=0
for f in "$@"; do
  LOG=/tmp/sweep_one.log
  MZ_OPTIMIZER_REBUILD=logical bin/sqllogictest -- "$f" > $LOG 2>&1
  TAIL=$(grep -E "^(    )?(PASS|FAIL)" $LOG | tail -1)
  if [[ "$TAIL" == *PASS* ]]; then
    PASS=$((PASS+1)); echo "- PASS $f" >> $OUT
  elif grep -q "panicked" $LOG; then
    ERR=$((ERR+1)); echo "- PANIC $f: $(grep -m1 'panicked at' $LOG | head -c 120)" >> $OUT
  else
    # Classify failure kinds: plan (EXPLAIN text) vs result rows.
    if grep -q "Explained Query" $LOG; then
      PLAN=$((PLAN+1))
      # arrangement-count comparison on first plan diff
      EXP=$(grep -A2 "expected:" $LOG | head -40 | grep -o "ArrangeBy" | wc -l | tr -d ' ')
      ACT=$(grep -A2 "actually:" $LOG | head -40 | grep -o "ArrangeBy" | wc -l | tr -d ' ')
      echo "- PLANDIFF $f ($TAIL) arrangements expected~$EXP actual~$ACT" >> $OUT
    else
      RESULT=$((RESULT+1)); echo "- RESULTDIFF $f ($TAIL)  <-- STOP-THE-LINE" >> $OUT
    fi
  fi
done
echo "" >> $OUT
echo "## Totals: pass=$PASS plandiff=$PLAN resultdiff=$RESULT panic=$ERR" >> $OUT
echo "sweep $NAME complete: pass=$PASS plandiff=$PLAN resultdiff=$RESULT panic=$ERR"
