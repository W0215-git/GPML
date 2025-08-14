import argparse
import json
import os

import wandb

# 获取当前工作路径
current_directory = os.getcwd()
print("current_directory:", current_directory)
# 拼接路径
config_path = os.path.join(current_directory, "config", "config.json")
print("config_path:", config_path)
def load_config(config_path=config_path):
    # 从 JSON 文件加载参数
    with open(config_path, "r") as f:
        config = json.load(f)

    # 创建 ArgumentParser 并从配置文件加载默认值
    parser = argparse.ArgumentParser(description="setting of MFGP")
    parser.add_argument("--project", type=str, default=config["project"],
                        help="project name")
    parser.add_argument("--dataDir", type=str, default=config["dataDir"],
                        help="Different processing strategy of data")
    parser.add_argument('--r', type=int, default=config["r"], help='randomSeeds')
    parser.add_argument('--Class', type=int, default=config["Class"], help='how many classes you want to classify:2 or 3')
    parser.add_argument("--population_size", type=int, default=config["population_size"], help="Population size")
    parser.add_argument("--generation", type=int, default=config["generation"], help="Number of generations")
    parser.add_argument("--cxProb", type=float, default=config["cxProb"], help="Crossover probability")
    parser.add_argument("--mutProb", type=float, default=config["mutProb"], help="Mutation probability")
    parser.add_argument("--elitismProb", type=float, default=config["elitismProb"], help="Elitism probability")
    parser.add_argument("--initialMinDepth", type=int, default=config["initialMinDepth"], help="Initial minimum depth")
    parser.add_argument("--initialMaxDepth", type=int, default=config["initialMaxDepth"], help="Initial maximum depth")
    parser.add_argument("--maxDepth", type=int, default=config["maxDepth"], help="Maximum depth")
    parser.add_argument("--tournament_size", type=int, default=config["tournament_size"], help="tournament_size")
    parser.add_argument("--Strategy", type=str, default=config["Strategy"], help="A parameter for decision-level fusion")
    parser.add_argument("--K", type=int, default=config["K"], help="K fold cross validation")
    parser.add_argument("--w1", type=float, default=config["w1"], help="the weight 1 of weighted fusion")
    parser.add_argument("--w2", type=float, default=config["w2"], help="the weight 2 of weighted fusion")
    parser.add_argument("--w3", type=float, default=config["w3"], help="the weight 3 of weighted fusion")
    parser.add_argument("--des", type=str, default=config["des"], help="Some important modification can be described here")

    # 解析参数
    args = parser.parse_args()
    return args


args = load_config()

config = {
    "project": args.project,
    "dataDir": args.dataDir,
    "random_seed": args.r,
    "Class": args.Class,
    "population_size": args.population_size,
    "generation": args.generation,
    "cxProb": args.cxProb,
    "mutProb": args.mutProb,
    "elitismProb": args.elitismProb,
    "initialMinDepth": args.initialMinDepth,
    "initialMaxDepth": args.initialMaxDepth,
    "maxDepth": args.maxDepth,
    "tournament_size": args.tournament_size,
    "Strategy": args.Strategy,
    "K": args.K,
    "w1": args.w1,
    "w2": args.w2,
    "w3": args.w3,
    "description": args.des

}

def setup_wandb():
    wandb.init(
        project=f"{config['project']}",
        name=f"{config['random_seed']}",
        config=config
    )
    return wandb