#!/usr/bin/env bash
# Delete any lingering eval endpoints (those with the eval- name prefix).
# Run this after any failed or interrupted eval run to stop billing.
set -euo pipefail

echo "Listing active endpoints..."
ENDPOINTS=$(nebius ai endpoint list --format json 2>/dev/null | \
  python3 -c "import sys, json; items=json.load(sys.stdin).get('items',[]); [print(i['metadata']['id']) for i in items if i['metadata']['name'].startswith('eval-')]" 2>/dev/null || true)

if [ -z "$ENDPOINTS" ]; then
  echo "No eval endpoints found — nothing to clean."
  exit 0
fi

echo "Deleting endpoints:"
while IFS= read -r ep_id; do
  echo "  deleting $ep_id"
  nebius ai endpoint delete --id "$ep_id"
done <<< "$ENDPOINTS"

echo "Done."
