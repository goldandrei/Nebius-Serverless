.PHONY: serve list-models clean

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

list-models:
	$(UVPY) -c "import sys; sys.path.insert(0,'.'); from src.eval_runner import list_catalog; list_catalog()"

clean:
	bash scripts/cleanup.sh
