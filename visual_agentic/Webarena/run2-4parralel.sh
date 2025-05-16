#!/bin/bash

set -e

PUBLIC_HOSTNAME="127.0.0.1"

SHOPPING_PORT=7770
SHOPPING_ADMIN_PORT=7780
REDDIT_PORT=9999
GITLAB_PORT=8023
WIKIPEDIA_PORT=8888
MAP_PORT=4000
HOMEPAGE_PORT=4399

export SHOPPING="http://${PUBLIC_HOSTNAME}:${SHOPPING_PORT}"
export SHOPPING_ADMIN="http://${PUBLIC_HOSTNAME}:${SHOPPING_ADMIN_PORT}/admin"
export REDDIT="http://${PUBLIC_HOSTNAME}:${REDDIT_PORT}"
export GITLAB="http://${PUBLIC_HOSTNAME}:${GITLAB_PORT}"
export WIKIPEDIA="http://${PUBLIC_HOSTNAME}:${WIKIPEDIA_PORT}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export MAP="http://${PUBLIC_HOSTNAME}:${MAP_PORT}"
export HOMEPAGE="http://${PUBLIC_HOSTNAME}:${HOMEPAGE_PORT}"

# export SHOPPING="http://metis.lti.cs.cmu.edu:7770"
# export SHOPPING_ADMIN="http://metis.lti.cs.cmu.edu:7780/admin"
# export REDDIT="http://metis.lti.cs.cmu.edu:9999"
# export GITLAB="http://metis.lti.cs.cmu.edu:8023"
# export MAP="http://metis.lti.cs.cmu.edu:3000"
# export WIKIPEDIA="http://metis.lti.cs.cmu.edu:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
# export HOMEPAGE="http://metis.lti.cs.cmu.edu:4399"

# export AZURE_ENDPOINT="https://ventus-research.openai.azure.com/"
# export OPENAI_API_KEY="ExUALvuFWsDBKK3mwu7g5finpGsPiPOfC4MnMWCYrUk82ZUR7p9YJQQJ99AKACYeBjFXJ3w3AAABACOGC9bJ"
# export ANTHROPIC_API_KEY="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"
# export GEMINI_API_KEY="<gemini_api_key>" # Optional, required when you run tasks with Gemini.

# python browser_env/auto_login.py

python eval_webarena_llama_all.py --config AgentOccam/configs/AgentOccam-llama-1.3.yml  --parallel 2