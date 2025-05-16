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

python browser_env/auto_login.py

python eval_webarena_llama_test.py --config AgentOccam/configs/Epoch.yml --num 9 
# export CUDA_VISIBLE_DEVICES=0
# python eval_webarena_llama1229.py --config AgentOccam/configs/AgentOccam-llama-1.3.yml --parallel 101 --num 0 >logs/llamanew/AgentOccam-llama-parallel-0103-101-0.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# python eval_webarena_llama1229.py --config AgentOccam/configs/AgentOccam-llama-1.3.yml --parallel 102 --num 0 >logs/llamanew/AgentOccam-llama-parallel-0103-102-0.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# python eval_webarena_llama1229.py --config AgentOccam/configs/AgentOccam-llama-1.3.yml --parallel 103 --num 0 >logs/llamanew/AgentOccam-llama-parallel-0103-103-0.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# python eval_webarena_llama1229.py --config AgentOccam/configs/AgentOccam-llama-1.3.yml --parallel 104 --num 0 >logs/llamanew/AgentOccam-llama-parallel-0103-104-0.log 2>&1 &


# echo "Waiting for previous tasks to complete..."
# while true; do
#     # 检查并等待所有之前启动的任务
#     running=$(pgrep -f "eval_webarena_llama1229.py.*--parallel (101|102|103|104)" | wc -l)
#     if [ "$running" -eq 0 ]; then
#         echo "All previous tasks have completed."
#         break
#     fi
#     sleep 60 # 每隔5秒检查一次
# done

# export CUDA_VISIBLE_DEVICES=0
# python eval_webarena_llama1229.py --config AgentOccam/configs/AgentOccam-llama-1.3.yml --parallel 105 --num 0 >logs/llamanew/AgentOccam-llama-parallel-0103-105-0.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=1
# python eval_webarena_llama1229.py --config AgentOccam/configs/AgentOccam-llama-1.3.yml --parallel 106 --num 0 >logs/llamanew/AgentOccam-llama-parallel-0103-106-0.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=2
# python eval_webarena_llama1229.py --config AgentOccam/configs/AgentOccam-llama-1.3.yml --parallel 107 --num 0 >logs/llamanew/AgentOccam-llama-parallel-0103-107-0.log 2>&1 &

# export CUDA_VISIBLE_DEVICES=3
# python eval_webarena_llama1229.py --config AgentOccam/configs/AgentOccam-llama-1.3.yml --parallel 108 --num 0 >logs/llamanew/AgentOccam-llama-parallel-0103-108-0.log 2>&1 &