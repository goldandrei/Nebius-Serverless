.PHONY: serve local-mock list-models up eval local down dashboard sweep clean

PKGS = pyyaml,openai,boto3,requests
UV   = uv run --with $(PKGS)
UVPY = uv run --with $(PKGS) python

# Load .env if it exists
ifneq (,$(wildcard .env))
  include .env
  export
endif

serve:
	$(UV) scripts/server.py

local-mock:
	$(UV) scripts/run_local_mock.py

list-models:
	$(UVPY) -c "import sys; sys.path.insert(0,'.'); from src.eval_runner import list_catalog; list_catalog()"

up:
	$(UVPY) -c "import sys; sys.path.insert(0,'.'); from src.orchestrator import create_all; create_all()"

eval:
	$(UVPY) -c "import json,sys; sys.path.insert(0,'.'); from src.eval_runner import run; r=run(mock=False); open('results/results.json','w').write(json.dumps(r,indent=2)); print('done — results/results.json')"

local:
	$(UVPY) -c "import json,sys; sys.path.insert(0,'.'); from src.eval_runner import run; r=run(backend='local'); open('results/results.json','w').write(json.dumps(r,indent=2)); print('done — results/results.json')"

down:
	$(UVPY) -c "import sys; sys.path.insert(0,'.'); from src.orchestrator import delete_all; delete_all()"

dashboard:
	$(UVPY) -c "import webbrowser,pathlib; webbrowser.open(pathlib.Path('dashboard/dashboard.html').resolve().as_uri())"

sweep:
	$(UV) src/sweep.py

clean:
	bash scripts/cleanup.sh
