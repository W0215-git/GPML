import random
import numpy as np
from deap import tools
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
from visualization.dataDistribution import visualize_tsne, visualize_tsne_whole_dataset
import time
import random

def varAndMultiTree(population, toolbox, cxpb, mutpb, modalities):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb = cxpb / (cxpb + mutpb)
    i = 1
    while i < len(offspring):
        if random.random() < new_cxpb:
            if offspring[i - 1] == offspring[i]:
                for j, modality in enumerate(modalities):
                    mutate_func = getattr(toolbox, f"mutate_{modality}")
                    mutate_func(offspring[i - 1][j])
                    mutate_func(offspring[i][j])
            else:
                for j, modality in enumerate(modalities):
                    mate_func = getattr(toolbox, f"mate_{modality}")
                    mate_func(offspring[i - 1][j], offspring[i][j])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i += 2
        else:
            for j, modality in enumerate(modalities):
                mutate_func = getattr(toolbox, f"mutate_{modality}")
                mutate_func(offspring[i][j])
            del offspring[i].fitness.values
            i += 1
    return offspring

def evalTrain(individual, wandb, toolbox, x_train, y_train):
    modalities = list(x_train.keys())
    num_modalities = len(modalities)
    individual_dict = {modality: individual[i] for i, modality in enumerate(modalities)}
    individual_dict["fusion"] = individual[-1]
    feature_extractors = {
        modality: getattr(toolbox, f"compile_{modality}")(expr=individual_dict[modality])
        for modality in modalities
    }
    fusion_func = getattr(toolbox, "compile_fusion")(expr=individual_dict["fusion"])
    scalers = {modality: MinMaxScaler() for modality in feature_extractors}
    scaler_fusion = MinMaxScaler()
    extracted_features = {
        modality: scalers[modality].fit_transform(
            np.asarray([feature_extractors[modality](x_train[modality][i, :, :]) for i in range(len(y_train))])
        )
        for modality in feature_extractors
    }
    combined_features = np.asarray([
        fusion_func(*[extracted_features[modality][i] for modality in extracted_features])
        for i in range(len(y_train))
    ])
    combined_features = scaler_fusion.fit_transform(combined_features)
    classifiers = {modality: SVC(kernel='linear', max_iter=1000, probability=True, random_state=i)
                   for i, modality in enumerate(extracted_features)}
    classifiers["fusion"] = SVC(kernel='linear', max_iter=1000, probability=True, random_state=len(extracted_features))
    skf = StratifiedKFold(n_splits=wandb.config['K'])
    all_true_labels = []
    all_predictions = []
    for train_idx, val_idx in skf.split(combined_features, y_train):
        train_features = {modality: extracted_features[modality][train_idx] for modality in extracted_features}
        val_features = {modality: extracted_features[modality][val_idx] for modality in extracted_features}
        train_features["fusion"], val_features["fusion"] = combined_features[train_idx], combined_features[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        if wandb.config['Strategy'] == "weight":
            for modality in classifiers:
                classifiers[modality].fit(train_features[modality], y_train_fold)
            prob_predictions = {
                modality: classifiers[modality].predict_proba(val_features[modality])
                for modality in classifiers
            }
            fused_prob = sum(wandb.config[f'w{modality}'] * prob_predictions[modality] for modality in classifiers)
            final_predictions = np.argmax(fused_prob, axis=1)
        elif wandb.config['Strategy'] == "voting":
            discrete_predictions = {
                modality: classifiers[modality].fit(train_features[modality], y_train_fold).predict(
                    val_features[modality])
                for modality in classifiers
            }
            final_predictions = []
            for i in range(len(y_val_fold)):
                votes = [discrete_predictions[modality][i] for modality in classifiers]
                final_predictions.append(max(set(votes), key=votes.count))
        elif wandb.config['Strategy'] == "single":
            classifiers["fusion"].fit(train_features["fusion"], y_train_fold)
            final_predictions = classifiers["fusion"].predict(val_features["fusion"])
        else:
            raise ValueError("Invalid strategy. Choose 'weight', 'voting', or 'single'.")
        all_true_labels.extend(y_val_fold)
        all_predictions.extend(final_predictions)
    accuracy = accuracy_score(all_true_labels, all_predictions)
    class_report = classification_report(all_true_labels, all_predictions, output_dict=True, zero_division=0)
    f1_score = round(class_report["weighted avg"]["f1-score"] * 100, 2)
    kappa = cohen_kappa_score(all_true_labels, all_predictions)
    metrics = {
        "accuracy": accuracy,
        "f1_score": f1_score,
        "kappa": kappa,
        "class_acc": {label: report["precision"] for label, report in class_report.items() if label.isdigit()},
    }
    individual.metrics = metrics
    return accuracy,

def evalTest(individual, wandb, toolbox, x_train, y_train, x_test, y_test):
    print("✅ Eval at Test set started")
    test_begin_time = time.process_time()
    try:
        modalities = list(x_train.keys())
        num_modalities = len(modalities)
        individual_dict = {modality: individual[i] for i, modality in enumerate(modalities)}
        individual_dict["fusion"] = individual[-1]
        feature_extractors = {
            modality: getattr(toolbox, f"compile_{modality}")(expr=individual_dict[modality])
            for modality in modalities
        }
        fusion_func = getattr(toolbox, "compile_fusion")(expr=individual_dict["fusion"])
        extracted_features_train = {
            modality: np.asarray(
                [feature_extractors[modality](x_train[modality][i, :, :]) for i in range(len(y_train))])
            for modality in feature_extractors
        }
        extracted_features_test = {
            modality: np.asarray([feature_extractors[modality](x_test[modality][i, :, :]) for i in range(len(y_test))])
            for modality in feature_extractors
        }
        combined_train_features = np.asarray([
            fusion_func(*[extracted_features_train[modality][i] for modality in extracted_features_train])
            for i in range(len(y_train))
        ])
        combined_test_features = np.asarray([
            fusion_func(*[extracted_features_test[modality][i] for modality in extracted_features_test])
            for i in range(len(y_test))
        ])
        scalers = {modality: MinMaxScaler() for modality in extracted_features_train}
        scaler_fusion = MinMaxScaler()
        extracted_features_train = {
            modality: scalers[modality].fit_transform(extracted_features_train[modality])
            for modality in extracted_features_train
        }
        extracted_features_test = {
            modality: scalers[modality].transform(extracted_features_test[modality])
            for modality in extracted_features_test
        }
        combined_train_features = scaler_fusion.fit_transform(combined_train_features)
        combined_test_features = scaler_fusion.transform(combined_test_features)
        classifiers = {modality: SVC(kernel='linear', max_iter=1000, probability=True, random_state=i)
                       for i, modality in enumerate(extracted_features_train)}
        classifiers["fusion"] = SVC(kernel='linear', max_iter=1000, probability=True,
                                    random_state=len(extracted_features_train))
        for modality in classifiers:
            classifiers[modality].fit(
                extracted_features_train[modality] if modality != "fusion" else combined_train_features, y_train)
        if wandb.config['Strategy'] == "weight":
            prob_predictions = {
                modality: classifiers[modality].predict_proba(
                    extracted_features_test[modality] if modality != "fusion" else combined_test_features)
                for modality in classifiers
            }
            fused_prob = sum(wandb.config[f'w{modality}'] * prob_predictions[modality] for modality in classifiers)
            final_predictions = np.argmax(fused_prob, axis=1)
        elif wandb.config['Strategy'] == "voting":
            discrete_predictions = {
                modality: classifiers[modality].predict(
                    extracted_features_test[modality] if modality != "fusion" else combined_test_features)
                for modality in classifiers
            }
            final_predictions = []
            for i in range(len(y_test)):
                votes = [discrete_predictions[modality][i] for modality in classifiers]
                final_predictions.append(max(set(votes), key=votes.count))
        elif wandb.config['Strategy'] == "single":
            final_predictions = classifiers["fusion"].predict(combined_test_features)
        else:
            raise ValueError("Invalid strategy. Choose 'weight', 'voting', or 'single'.")
        test_end_time = time.process_time()
        test_duration = test_end_time - test_begin_time
        accuracy = round(100 * np.mean(final_predictions == y_test), 4)
        class_report = classification_report(y_test, final_predictions, output_dict=True, zero_division=0)
        f1_score = round(class_report["weighted avg"]["f1-score"] * 100, 4)
        kappa = round(cohen_kappa_score(y_test, final_predictions), 4)
        class_acc = {label: round(report["precision"] * 100, 4) for label, report in class_report.items() if label.isdigit()}
        print("Eval at Test set finished successfully")
        print("Start feature distribution visualization")
        print(f"====================== Test Results ==========================")
        print(f" - Best individual contains feature dimensions per modality:")
        for modality in modalities:
            print(f"   - {modality}: {extracted_features_train[modality].shape[1]} features")
        print(f" - Combined Features: {combined_train_features.shape[1]}")
        print(f" - Accuracy at Test set: {accuracy}%")
        print(f" - F1 Score at Test set: {f1_score}")
        print(f" - Kappa at Test set: {kappa}")
        print(f" - Test time (hours): {test_duration / 3600}")
        for i in range(wandb.config["Class"]):
            print(f" - Class-{i} Accuracy at Test set: {class_acc.get(f'{i}', 'N/A')}")
        log_data = {
            "Feature Dimensions": {modality: extracted_features_train[modality].shape[1] for modality in modalities},
            "Combined Features": combined_train_features.shape[1],
            "Acc at Test set": accuracy,
            "F1 at Test set": f1_score,
            "Kappa at Test set": kappa,
            "Test Time (min)": test_duration / 60,
            "final_predictions": final_predictions
        }
        for modality in modalities:
            log_data[f"{modality} Tree"] = str(individual_dict[modality])
            log_data[f"Depth({modality})"] = str(individual_dict[modality].height)
            log_data[f"Size({modality})"] = str(len(individual_dict[modality]))
        log_data["Fusion Tree"] = str(individual_dict["fusion"])
        log_data["Depth(Fusion)"] = str(individual_dict["fusion"].height)
        log_data["Size(Fusion)"] = str(len(individual_dict["fusion"]))
        for i in range(wandb.config["Class"]):
            log_data[f"Class-{i} Accuracy at Test set"] = class_acc.get(f"{i}", 'N/A')
        wandb.log(log_data)
        print(f"====================== wandb saved ==========================")
        return {
            "feature_dimensions": {modality: extracted_features_train[modality].shape[1] for modality in modalities},
            "feature_dim_combined": combined_train_features.shape[1],
            "accuracy": accuracy,
            "f1_score": f1_score,
            "kappa": kappa,
            "class_accuracy": class_acc
        }
    except Exception as e:
        print(f"❌ Error in evalTest: {str(e)}")
        return {
            "feature_dimensions": {},
            "feature_dim_combined": 0,
            "accuracy": 0,
            "f1_score": 0,
            "kappa": 0,
            "class_accuracy": {}
        }

def eaSimple(modalities, population, toolbox, cxpb, mutpb, elitepb, ngen, halloffame=None, verbose=__debug__,
             progress_bar=None, wandb=None):
    logbook = tools.Logbook()
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_f1 = tools.Statistics(key=lambda ind: ind.metrics["f1_score"])
    stats_kappa = tools.Statistics(key=lambda ind: ind.metrics["kappa"])
    stats_size = {modality: tools.Statistics(key=lambda ind: len(ind[i])) for i, modality in enumerate(modalities)}
    stats_size["fusion"] = tools.Statistics(key=lambda ind: len(ind[-1]))
    stats_depth = {modality: tools.Statistics(key=lambda ind: ind[i].height) for i, modality in enumerate(modalities)}
    stats_depth["fusion"] = tools.Statistics(key=lambda ind: ind[-1].height)
    if wandb.config["Class"] == 2:
        labels = ["0", "1"]
    elif wandb.config["Class"] == 3:
        labels = ["0", "1", "2"]
    else:
        raise ValueError(f"Unsupported class type: {wandb.config['Class']}. Please check the configuration.")
    class_acc_stats = {label: tools.Statistics(key=lambda ind: ind.metrics["class_acc"].get(label, 0)) for label in labels}
    mstats = tools.MultiStatistics(
        fitness=stats_fit,
        f1_score=stats_f1,
        kappa=stats_kappa,
        **{f"size_{modality}": stats_size[modality] for modality in stats_size},
        **{f"depth_{modality}": stats_depth[modality] for modality in stats_depth},
        **{f"class_acc_{label}": stat for label, stat in class_acc_stats.items()}
    )
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("max", np.max)
    logbook.header = ["gen", "evals"] + mstats.fields
    print("\nLogbook header:", logbook.header)
    print("Calculating fitness for the initial population.")
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.mapp(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    if halloffame is not None:
        halloffame.update(population)
    for gen in range(1, ngen + 1):
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.write(f"\n----- Generation {gen} -----")
        elitism_num = int(elitepb * len(population))
        offspring = toolbox.select(population, len(population) - elitism_num)
        offspring = list(map(toolbox.clone, offspring))
        offspring = varAndMultiTree(offspring, toolbox, cxpb, mutpb, modalities)
        elites = tools.selBest(population, elitism_num)
        offspring.extend(elites)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        if invalid_ind:
            fitnesses = toolbox.mapp(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        population[:] = offspring
        if halloffame is not None:
            halloffame.update(population)
        record_raw = mstats.compile(population) if mstats else {}
        record = {
            "fitness": {k: record_raw["fitness"][k] for k in ["avg", "std", "max"]},
            "f1_score": {k: record_raw["f1_score"][k] for k in ["avg", "std", "max"]},
            "kappa": {k: record_raw["kappa"][k] for k in ["avg", "std", "max"]},
            **{f"size_{modality}": {k: record_raw[f"size_{modality}"][k] for k in ["avg", "std", "max"]} for modality in stats_size},
            **{f"depth_{modality}": {k: record_raw[f"depth_{modality}"][k] for k in ["avg", "std", "max"]} for modality in stats_depth},
            "class_acc": {
                label: {k: record_raw[f"class_acc_{label}"][k] for k in ["avg", "std", "max"]}
                for label in labels
            },
        }
        logbook.record(**record)
        if verbose:
            print(f"Generation {gen}: {record}")
        if wandb is not None:
            wandb.log(record)
    return population, logbook
