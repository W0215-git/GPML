import time
import operator
from functools import partial
from load_data import getData
from tqdm import tqdm
import evalGP_main as evalGP
from deap import base, creator, tools
from config.config_loader import setup_wandb
from defineIndividial import *

wandb = setup_wandb()
modalities, x_train, x_test, y_train, y_test = getData(wandb)

pset_list = {}
for modality in modalities:
    bound1, bound2, channel_count = x_train[modality][1].shape
    if channel_count == 1:
        pset_list[modality] = create_pset_gray(modality, bound1, bound2)
        print(f"✅ Created Gray-Scale GP tree for modality: {modality} (1 channel)")
    elif channel_count == 3:
        pset_list[modality] = create_pset_color(modality, bound1, bound2)
        print(f"✅ Created Color-Scale GP tree for modality: {modality} (3 channel)")
    else:
        pset_list[modality] = create_pset_multic(modality, bound1, bound2, channel_count)
        print(f"✅ Created Multi-Channel GP tree for modality: {modality} ({channel_count} channels)")

pset_fusion = create_pset_fusion(len(modalities))

print(f"\n🌳 Total GP trees created: {len(pset_list)}")
for modality, tree in pset_list.items():
    print(f" - {modality}: {tree}")

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_trees", initTrees,
                 pset_list=pset_list, pset_fusion=pset_fusion,
                 min_depth=wandb.config['initialMinDepth'], max_depth=wandb.config['initialMaxDepth'])
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_trees)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

for modality, pset in pset_list.items():
    toolbox.register(f"compile_{modality}", gp.compile, pset=pset)
toolbox.register("compile_fusion", gp.compile, pset=pset_fusion)

toolbox.register("mapp", map)
toolbox.register("evaluate", partial(evalGP.evalTrain, wandb=wandb, toolbox=toolbox,
                                     x_train=x_train, y_train=y_train))
toolbox.register("select", tools.selTournament, tournsize=wandb.config['tournament_size'])
toolbox.register("selectElitism", tools.selBest)

for modality in pset_list.keys():
    toolbox.register(f"mate_{modality}", gp.cxOnePoint)
toolbox.register("mate_fusion", gp.cxOnePoint)

for modality in pset_list.keys():
    toolbox.register(f"expr_mut_{modality}", gp_restrict.genFull, min_=0, max_=2)
    toolbox.register(f"mutate_{modality}", gp.mutUniform, expr=toolbox.__getattribute__(f"expr_mut_{modality}"), pset=pset_list[modality])
toolbox.register("expr_mut_fusion", gp_restrict.genFull, min_=0, max_=2)
toolbox.register("mutate_fusion", gp.mutUniform, expr=toolbox.expr_mut_fusion, pset=pset_fusion)

for modality in pset_list.keys():
    toolbox.decorate(f"mate_{modality}", gp.staticLimit(key=operator.attrgetter("height"), max_value=wandb.config['maxDepth']))
    toolbox.decorate(f"mutate_{modality}", gp.staticLimit(key=operator.attrgetter("height"), max_value=wandb.config['maxDepth']))
toolbox.decorate("mate_fusion", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))
toolbox.decorate("mutate_fusion", gp.staticLimit(key=operator.attrgetter("height"), max_value=4))

def GPMain(randomSeeds):
    random.seed(randomSeeds)
    print("GPMain started with random seed: ", randomSeeds)
    print("Generating initial population...")
    population = toolbox.population(n=wandb.config['population_size'])
    hof = tools.HallOfFame(10)
    print("Starting evolution process...")
    begin_time = time.process_time()
    with tqdm(total=wandb.config['generation'], desc="Generation Progress") as pbar:
        population, log = evalGP.eaSimple(
            modalities=modalities,
            population=population,
            toolbox=toolbox,
            cxpb=wandb.config['cxProb'],
            mutpb=wandb.config['mutProb'],
            elitepb=wandb.config['elitismProb'],
            ngen=wandb.config['generation'],
            halloffame=hof,
            verbose=True,
            progress_bar=pbar,
            wandb=wandb
        )
    print("Evolution process completed.")
    end_time = time.process_time()
    train_duration = end_time - begin_time
    wandb.log({
        "Training Time (hours)": train_duration / 3600
    })
    print(f" - Training time (hours): {train_duration / 3600}")
    return population, log, hof

if __name__ == "__main__":
    population, log, hof = GPMain(wandb.config['random_seed'])
    test_results = evalGP.evalTest(
        hof[0], wandb, toolbox, x_train, y_train, x_test, y_test
    )
    wandb.finish()
    print("End of the experiment.")
