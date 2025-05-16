import argparse
import os
import re
import shutil
import time

import ipdb
from AgentOccam.AgentOccam_llama import AgentOccam
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
    parser.add_argument(
        "--config", type=str, required=True, help="yaml config file location"
    )
    parser.add_argument(
        "--parallel", type=int, required=True, help="Parallel execution number"
    )
    parser.add_argument(
        "--num", type=int, required=True, help="num"
    )
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = DotDict(yaml.safe_load(file))

    if config.logging:
        if config.logname:
            config.logname = config.logname + str(args.num)
            dstdir = f"{config.logdir}/{config.logname}"
        else:
            dstdir = f"{config.logdir}/{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(dstdir, exist_ok=True)
        shutil.copyfile(args.config, os.path.join(dstdir, args.config.split("/")[-1]))
    random.seed(81)

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
        
        99: [680,700,701,702,771,772,773,774,781,782,790],
        
        44: [97, 265, 266, 267, 268, 424, 425, 426, 427, 428, 429, 430, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 671, 672, 673, 674, 675, 681, 682, 737, 738, 739, 740, 741,759, 760, 791],
        
        #-----
        
        12: [ 0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 41, 42, 43, 62, 63, 64, 65, 77, 78, 79, 94, 95, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 119, 120, 121, 122, 123, 127, 128, 129,130, 131,157,183,184, 185, 186, 187, 193, 194, 195, ],
        
        21: [ 214, 215, 216,217, 243, 244, 245,  246, 247, 288, 289, 290, 291, 292, 344, 345, 346, 347,348, 374,375, 423,453,454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 470, 471, 472, 473, 474, 486, 487, 488, 489, ],
        
        31: [  490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500,  501, 502, 503, 504, 505, 538, 539, 540,541, 542, 543, 544,545, 546, 547, 548, 549, 550, 551, 676, 677, 678, 679, 680, 694, 695, 696, 697, 698, 699, 700, 701, ],
        
        41: [702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 768, 769, 770, 771, 772, 773, 774, 775,776, 777, 778, 779, 780, 781, 782, 790, 196, 197, 198, 199, 200, 201, 202, 203, 204, 208, 209, 210, 211, 212, 213,], 
        
        # 12:[  202, 203, 204,],
        # 21:[ 677, 678, ],
        # 31:[199, 200, 201,],
        # 41:[679, 680, 694],
        
        #-----
        
        13: [ 7, 8, 9, 10, 16, 17, 18, 19, 20, 32, 33, 34, 35, 36, 37, 38, 39, 40, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 70, 71, 72, 73, 74, 75, 76, 80,81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 98, 99, 100, 101, 137, 138, 139, 140, 151,],
        
        22: [ 152, 153, 154, 155, 218, 219, 220, 221, 222, 223, 224, 236, 237, 248, 249, 250, 251, 252, 253, 254, 255, 256,257, 287, 356, 363, 364, 365, 366, 367, 369, 370, 371, 372, 373, 377, 378, 379, 380, 381, 382, 383, 757, 758, 761, 762, 763, 764, 765, 766, 767],
        
    #-----
        
        42: [ 21, 22, 23, 24, 25, 26, 47, 48, 49, 50, 51, 96, 117, 118, 124,125, 126, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 188, 189, 190, 191, 192, 225, 226, 227, 228, 229, 230, 231,232, 233, 234, 235, 238, 239, 240, 241, 242, 260,261, 262, 263, 264, 269,  ],
        
        
        
        32: [ 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 298, 299, 300, 301, 302, 313, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328,329, 330, 331, 332, 333, 334,335, 336, 337, 338, 351, 352, 353, 354, 355, 358, 359, 360, 361, 362, 368, 376, 384, 385, 386, 387, 388, 431, ],
        
        14: [467, 468, 469, 506, 507, 508, 509, 510,511, 512, 513,514, 515, 516, 517, 518, 519, 520, 521, 528, 529, 530, 531, 532, 571, 572, 573, 574, 575, 585, 586, 587, 588, 589, 653, 654, 655, 656, 657, 689, 690, 691, 692, 693, 792, 793, 794, 795, 796, 797, 798],
        
        23: [617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633,684,685,270, 271, 272, 273,432, 433, 434, 435, 436, 437, 438, 439, 440, 465, 466, ],
        
        
          # --------
        24: [ 44, 45, 46, 102,103, 104, 105, 106, 132, 133, 134, 135, 136, 156, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 205, 206, 207, 258, 259, 293, 294, 295, 296, 297, 303, 304, 305, 306, 307, 308, 309, 310, 311,312, 314, 315, 316, 317, 318, 339, 340, 341, 342, 343, 349, 350, 357, 389, 390, 391, 392, 393,],
        
        43: [ 394, 395, 396, 397, 398, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 441, 442, 443, 444, 445, 446, 447, 448,449, 450, 451, 452, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 522, 523, 524, 525, 526, 527, 533, 534, 535, 536, 537, 567, 568, 569, 570, 576, 577, 578, 579, 590, 591, 592, 593, 594, ],
        
        
        33: [658, 659, 660, 661, 662,663, 664, 665, 666, 667, 668, 669, 670, 736, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 783, 784, 785, 786, 787, 788, 789, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811],
        
        11: [600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616,681,682,683, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649,686,687,688],
        # --------
        34: [27, 28, 29, 30, 31, 66, 67, 68, 69, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 580, 581, 582, 583, 584, 595, 596, 597, 598, 599,650, 651, 652, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735],
        
        211: [323, 324, 325, 326, 327, 328,329, 330, 331, 332, 333,],
        221: [ 334,335, 336, 337, 338, 351, 352, 353, 354, 355, 358, ],
        231: [359, 360, 361, 362, 368, 376, 384, 385, 386, 387, 388, ],
        241: [ 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 465, 466, ],
        
        212:[694, 695, 696, 697, 698, 699, 700, 701, 702, ],
        222:[703, 704, 705, 706, 707, 708, 709, 710, 711, ],
        232:[712, 713, 768, 769, 770, 771, 772, 773, 774, ],
        242:[ 775,776, 777, 778, 779, 780, 781, 782, 790,],
        
        
        101: [5, 78, 77, 80, 71, 76, 47, 36, 6, 10, 34, 51, 42],
        102: [150, 180, 109, 186, 131, 135, 165, 112, 142, 115, 163, 125, 184, 140, 156, 182],
        103: [299, 294, 271, 242, 248, 254, 283, 223, 228, 243, 272, 218, 202, 290],
        104: [312, 318, 304, 309, 366, 301, 354, 322, 351, 353, 347],
        105: [478, 383, 476, 465, 462, 382, 461, 419, 422, 452, 432,  442, 445, 412],
        106: [570, 544, 518, 490, 484, 538, 532, 546, 528, 550, 491, 563],
        107: [ 586,  665, 668, 675],
        108: [478, 383, 476, 465, 462, 382, 461, 419, 422, 452, 432, 406, 442, 445, 412],
        
        777:[322,780,781,782,790]
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
        raise ValueError(
            f"Invalid task range format for parallel value: {args.parallel}"
        )

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
            time.sleep(1200)
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
