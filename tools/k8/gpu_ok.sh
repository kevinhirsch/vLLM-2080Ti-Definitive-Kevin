#!/bin/bash
# exit 0 only if no planned-offline window is open
curl -s -m 5 localhost:8000/gateway/capacity | python3 -c 'import sys,json; d=json.load(sys.stdin); sys.exit(1 if d.get("planned_offline") else 0)'
