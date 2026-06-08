#!/bin/bash
# Compare THIS node's onpolicy/ python files against anjuna2's reference manifest
# (anjuna2 = the node where combo2 reached 63% / combo2_8M is running).
# Run from the repo root, e.g.:  bash tools/compare_to_anjuna2.sh
#
# Reference: branch feat/pursuit-checkpoint-resume, code commit 61dba21
# (github.com/yashrb24/on-policy). RESULT: IDENTICAL means this node runs the
# exact training code anjuna2 ran.

set -u
MAN="tools/manifest_anjuna2.sha256"
[ -f "$MAN" ] || { echo "ERROR: $MAN not found. Are you in the repo root on branch feat/pursuit-checkpoint-resume?"; exit 2; }

echo "node:      $(hostname)   repo: $(pwd)"
echo "git:       branch=$(git branch --show-current 2>/dev/null || echo '?')  HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "reference: anjuna2 / feat/pursuit-checkpoint-resume (code 61dba21)"
echo "----"

ok=0; differ=0; miss=0
while read -r h p; do
  if [ ! -f "$p" ]; then echo "MISSING  $p"; miss=$((miss+1)); continue; fi
  ah=$(sha256sum "$p" | cut -d' ' -f1)
  if [ "$ah" = "$h" ]; then ok=$((ok+1)); else echo "DIFFER   $p"; differ=$((differ+1)); fi
done < "$MAN"

# files present on this node but NOT in the reference (e.g. stale main/ leftovers from an rsync-over-main)
extra=$(comm -23 <(find onpolicy -name '*.py' 2>/dev/null | sort) <(awk '{print $2}' "$MAN" | sort))
[ -n "$extra" ] && echo "$extra" | sed 's/^/EXTRA    /'
nextra=$(printf '%s\n' "$extra" | grep -c . )

echo "----"
echo "match=$ok  differ=$differ  missing=$miss  extra=$nextra  (reference total $((ok+differ+miss)))"
if [ "$differ" -eq 0 ] && [ "$miss" -eq 0 ]; then
  echo "RESULT: onpolicy/ is IDENTICAL to anjuna2${extra:+ (note: $nextra extra unused .py files present, harmless)}"
else
  echo "RESULT: DRIFT DETECTED — $differ differing, $miss missing files vs anjuna2"
fi
