import argparse
import os
import re
import shutil
import time

import ipdb
from AgentOccam.AgentOccam_rag1203 import AgentOccam
from AgentOccam.env import WebArenaEnvironmentWrapper
from AgentOccam.prompts import AgentOccam_prompt
from AgentOccam.utils import EVALUATOR_DIR
from webagents_step.agents.step_agent import StepAgent
from webagents_step.prompts.webarena import (
    step_fewshot_template,
    step_fewshot_template_adapted,
)
from webagents_step.utils.data_prep import *


def run():
    parser = argparse.ArgumentParser(
        description="Only the config file argument should be passed"
    )
    parser.add_argument("--config", type=str, required=True, help="yaml config file location")
    parser.add_argument("--parallel", type=int, required=True, help="Parallel execution number")    
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = DotDict(yaml.safe_load(file))

    if config.logging:
        if config.logname:
            dstdir = f"{config.logdir}/{config.logname}"
        else:
            dstdir = f"{config.logdir}/{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(dstdir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(dstdir, args.config.split("/")[-1]))
    random.seed(42)

    config_file_list = []
    task_ranges = {
        1: (0, 100),   
        2: (100, 200),  
        3: (200, 300), 
        4: (300, 380), 
        5: (380, 480),  
        6: (480, 580),  
        7: (580, 680),  
        8: (680, 770),  
        9: (770, 813),
        99:[
        10,
        705,
        121,
        606,
        117,
        466,
        620,
        796,
        431,
        221,
        206,
        447,
        427,
        435,
        732,
        251,
        691,
        417,
        531,
        318,
        654,
        300,
        194,
        299,
        748,
        104,
        775,
        148,
        441,
        536,
        682,
        609,
        724,
        717,
        153,
        3,
        548,
        520,
        554,
        787,
        240,
        80,
        527,
        477,
        217,
        412,
        619,
        626,
        562,
        384,
        348,
        462,
        346,
        401,
        265,
        692,
        703,
        801,
        809,
        794,
        180,
        686,
        323,
        326,
        440,
        13,
        216,
        190,
        311,
        496,
        309,
        319,
        670,
        54,
        663,
        722,
        238,
        334,
        627,
        237,
        48,
        378,
        652,
        387,
        746,
        332,
        72,
        768,
        558,
        694,
        595,
        388,
        414,
        779,
        266,
        269,
        551,
        139,
        773,
        608
    ]
        

  
    }

    if args.parallel not in task_ranges:
        raise ValueError(f"Invalid parallel value: {args.parallel}")

    task_ids = task_ranges[args.parallel]


    if isinstance(task_ids, tuple) and len(task_ids) == 2:
        start, end = task_ids
        task_ids = list(range(start, end))
    elif isinstance(task_ids, list):
        task_ids = task_ids
    else:
        raise ValueError(f"Invalid task range format for parallel value: {args.parallel}")
    
    
    
    
    if hasattr(config.env, "relative_task_dir"):
        relative_task_dir = config.env.relative_task_dir
    else:
        relative_task_dir = "tasks"
    if task_ids == "all" or task_ids == ["all"]:
        task_ids = [
            filename[: -len(".json")]
            for filename in os.listdir(f"config_files/{relative_task_dir}")
            if filename.endswith(".json")
        ]
    for task_id in task_ids:
        config_file_list.append(f"config_files/{relative_task_dir}/{task_id}.json")

    fullpage = config.env.fullpage if hasattr(config.env, "fullpage") else True
    current_viewport_only = not fullpage

    if config.agent.type == "AgentOccam":
        agent_init = lambda: AgentOccam(
            prompt_dict={
                k: v
                for k, v in AgentOccam_prompt.__dict__.items()
                if isinstance(v, dict)
            },
            config=config.agent,
        )
    elif config.agent.type == "AgentOccam-SteP":
        agent_init = lambda: StepAgent(
            root_action=config.agent.root_action,
            action_to_prompt_dict={
                k: v
                for k, v in step_fewshot_template_adapted.__dict__.items()
                if isinstance(v, dict)
            },
            low_level_action_list=config.agent.low_level_action_list,
            max_actions=config.env.max_env_steps,
            verbose=config.verbose,
            logging=config.logging,
            debug=config.debug,
            model=config.agent.model_name,
            prompt_mode=config.agent.prompt_mode,
        )
    elif config.agent.type == "SteP-replication":
        agent_init = lambda: StepAgent(
            root_action=config.agent.root_action,
            action_to_prompt_dict={
                k: v
                for k, v in step_fewshot_template.__dict__.items()
                if isinstance(v, dict)
            },
            low_level_action_list=config.agent.low_level_action_list,
            max_actions=config.env.max_env_steps,
            verbose=config.verbose,
            logging=config.logging,
            debug=config.debug,
            model=config.agent.model_name,
            prompt_mode=config.agent.prompt_mode,
        )
    else:
        raise NotImplementedError(f"{config.agent.type} not implemented")

    for config_file in config_file_list:
        with open(config_file, "r") as f:
            task_config = json.load(f)
            print(f"Task {task_config['task_id']}.")
        if os.path.exists(os.path.join(dstdir, f"{task_config['task_id']}.json")):
            print(f"Skip {task_config['task_id']}.")
            continue
        if task_config["task_id"] in list(range(600, 650)) + list(range(681, 689)):
            print("Reddit post task. Sleep 30 mins.")
            time.sleep(1800)
        env = WebArenaEnvironmentWrapper(
            config_file=config_file,
            max_browser_rows=config.env.max_browser_rows,
            max_steps=config.max_steps,
            slow_mo=1,
            observation_type="accessibility_tree",
            current_viewport_only=current_viewport_only,
            viewport_size={"width": 1920, "height": 1080},
            headless=config.env.headless,
            global_config=config,
        )

        agent = agent_init()
        objective = env.get_objective()
        status = agent.act(objective=objective, env=env)
        env.close()

        if config.logging:
            with open(config_file, "r") as f:
                task_config = json.load(f)
            log_file = os.path.join(dstdir, f"{task_config['task_id']}.json")
            log_data = {
                "task": config_file,
                "id": task_config["task_id"],
                "model": (
                    config.agent.actor.model
                    if hasattr(config.agent, "actor")
                    else config.agent.model_name
                ),
                "type": config.agent.type,
                "trajectory": agent.get_trajectory(),
            }
            summary_file = os.path.join(dstdir, "summary.csv")
            summary_data = {
                "task": config_file,
                "task_id": task_config["task_id"],
                "model": (
                    config.agent.actor.model
                    if hasattr(config.agent, "actor")
                    else config.agent.model_name
                ),
                "type": config.agent.type,
                "logfile": re.search(r"/([^/]+/[^/]+\.json)$", log_file).group(1),
            }
            if status:
                summary_data.update(status)
            log_run(
                log_file=log_file,
                log_data=log_data,
                summary_file=summary_file,
                summary_data=summary_data,
            )


if __name__ == "__main__":
    run()
