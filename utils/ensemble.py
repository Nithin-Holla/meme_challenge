import argparse
import csv
import os
from itertools import product
from glob import glob
import logging
import numpy as np
from copy import copy
import random
from random import shuffle


logging.basicConfig(format='%(asctime)s : %(levelname)s - %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger('EnsembleLogger')

try:
    from data.metrics import aucroc, find_optimal_threshold
except ModuleNotFoundError: # For local usage
    import sys
    sys.path.append("../")
    from data.metrics import aucroc, find_optimal_threshold

try:
    from deap import creator, base, tools, algorithms
    EA_IMPORTED = True
except ModuleNotFoundError:
    logger.warning("Could not import DEAP library for optimizing ensemble weights. Consider installing it for better ensemble optimization  (pip install deap).")
    EA_IMPORTED = False




def find_ensemble(dev_files, test_files, weight_range=(0.0, 0.5, 1.0, 2.0), max_weights=10000):
    dev_preds = [load_csv(f) for f in dev_files]
    dev_preds = align_ids(dev_preds)
    dev_gt = dev_preds[0]["gt"]
    dev_scores = [aucroc(d["orig"]["proba"], d["orig"]["gt"]) for d in dev_preds]
    logger.info("Individual scores: " + ", ".join(["%4.2f%%" % (100.0*s) for s in dev_scores]))

    output_dir = dev_files[0].rsplit("/",1)[0]
    if dev_files[0].endswith("_00_preds.csv"):
        dev_name = "_".join(dev_files[0].rsplit("_",4)[-4:-1])
        model_name = dev_files[0].split("/")[-1].rsplit("_",6)[0]
    else:
        dev_name = "_".join(dev_files[0].rsplit("_",3)[-3:-1])
        model_name = dev_files[0].split("/")[-1].rsplit("_",5)[0]
    logger.info("Model name: %s" % model_name)

    predictions = np.stack([d["proba"] for d in dev_preds], axis=0)
    
    def eval_func(weights, on_logits=True):
        preds = create_ensemble_prediction(predictions=predictions,
                                           weights=weights,
                                           on_logits=on_logits)
        return float(aucroc(preds, dev_gt)),

    best_score, best_config = brute_force_finder(eval_func=eval_func,
                                                 num_weights=len(dev_preds), 
                                                 weight_range=weight_range, 
                                                 max_weights=max_weights)
    if EA_IMPORTED:
        logger.info("Starting EA to find optimal weights...")
        ea_score, ea_config = EA_ensemble_finder(eval_func=eval_func,
                                                 num_weights=len(dev_preds),
                                                 individual_scores=dev_scores)
        if ea_score > best_score:
            logger.info("Found better config with EA: [%s], on_logits=%s" % (", ".join(["%4.3f" % w for w in ea_config['weights']]), str(ea_config['on_logits'])))
            best_score = ea_score
            best_config = ea_config

    best_dict = copy(dev_preds[0])
    best_dict["proba"] = create_ensemble_prediction(predictions=predictions,
                                                    weights=best_config["weights"],
                                                    on_logits=best_config["on_logits"])
    logger.info("Finished search for ensemble weights.")

    threshold = find_optimal_threshold(best_dict["proba"], dev_gt)
    logger.info("Using threshold %4.3f for discrete predictions.\n" % threshold)

    best_dict["label"] = (best_dict["proba"] > threshold).astype(np.int32)
    export_csv(csv_dict=best_dict, 
               csv_file=os.path.join(output_dir, model_name + "_" + dev_name + "_ensemble.csv"))
    best_acc = (best_dict["label"] == dev_gt).mean()
    logger.info("Best score on %s: %4.2f%% (accuracy=%4.2f%%)\n" % (dev_name, best_score*100.0, best_acc*100.0))

    if not isinstance(test_files[0], list):
        test_files = [test_files]

    for test_list in test_files:
        test_name = "_".join(test_list[0].rsplit("_",3)[-3:-1])
        logger.info("Exporting %s ensemble..." % test_name)
        test_model_name = test_list[0].split("/")[-1].rsplit("_",5)[0]
        if model_name != test_model_name:
            logger.warning("Model name of test and dev did not fit (dev: %s, test: %s). Are you sure you have the correct files? Using %s for test export..." % (model_name, test_model_name, test_model_name))
        test_preds = [load_csv(f) for f in test_list]
        preds = create_ensemble_prediction(predictions=[d["proba"] for d in test_preds],
                                           weights=best_config["weights"],
                                           on_logits=best_config["on_logits"])
        test_dict = copy(test_preds[0])
        test_dict["proba"] = preds
        test_dict["label"] = (preds > threshold).astype(np.int32)
        if "gt" in test_dict:
            score = aucroc(test_dict["proba"], test_dict["gt"])
            acc = (test_dict["label"] == test_dict["gt"]).mean()
            logger.info("New ensemble score on %s: %4.2f%% (accuracy=%4.2f%%)" % (test_name, score*100.0, acc*100.0))
            test_scores = [aucroc(d["proba"], test_dict["gt"]) for d in test_preds]
            logger.info("Individual scores on %s have been: " % test_name + ", ".join(["%4.2f%%" % (100.0*s) for s in test_scores]))
            
        export_csv(csv_dict=test_dict, 
                   csv_file=os.path.join(output_dir, test_model_name + "_" + test_name + "_ensemble.csv"))


def load_csv(csv_file):
    with open(csv_file, 'r', newline='') as f:
        file_reader = csv.reader(f, delimiter=',')
        rows = list(file_reader)
    csv_dict = {k: [] for k in rows[0]}
    for column_index, column in enumerate(rows[0]):
        csv_dict[column] = [r[column_index] for r in rows[1:]]
        if column in ["proba"]:
            csv_dict[column] = [float(v) for v in csv_dict[column]]
        else:
            csv_dict[column] = [int(v) for v in csv_dict[column]]
        csv_dict[column] = np.array(csv_dict[column])
    return csv_dict


def align_ids(csv_dicts):
    all_ids = np.array(sorted(list(set([e for d in csv_dicts for e in d["id"].tolist()])))).squeeze()
    labels = [[d["gt"][np.where(d["id"] == data_id)[0]][0] for d in csv_dicts if data_id in d["id"]] for data_id in all_ids]
    assert all([all([llist[0] == l for l in llist]) for llist in labels]), "Label mismatch in the predictions. That should normally not happen. Something must be wrong with the predictions."
    labels = np.array([llist[0] for llist in labels]).squeeze()
    csv_dicts = [{"orig": d, "id": all_ids, "gt": labels} for d in csv_dicts]
    for d in csv_dicts:
        proba = [-1 if data_id not in d["orig"]["id"] else d["orig"]["proba"][np.where(d["orig"]["id"] == data_id)[0]][0] for data_id in all_ids]
        preds = [-1 if data_id not in d["orig"]["id"] else d["orig"]["label"][np.where(d["orig"]["id"] == data_id)[0]][0] for data_id in all_ids]
        d["proba"] = np.array(proba).squeeze()
        d["label"] = np.array(preds).squeeze()
    return csv_dicts


def export_csv(csv_dict, csv_file):
    if "orig" in csv_dict:
        _ = csv_dict.pop("orig")
    header = list(csv_dict.keys())
    s = ",".join(header)+"\n"
    rows = [[csv_dict[key][i] for key in header] for i in range(len(csv_dict[header[0]]))]
    for r in rows:
        s += ",".join([ ("%f" % e) if isinstance(e, float) else ("%i" % e)  for e in r])
        s += "\n"
    with open(csv_file, "w") as f:
        f.write(s)


def create_ensemble_prediction(predictions, weights, on_logits=False):
    if isinstance(predictions, list):
        predictions = np.stack(predictions, axis=0)
    if isinstance(weights, list) or isinstance(weights, tuple):
        weights = np.array(weights)

    inv_mask = (predictions == -1)
    predictions[inv_mask] = 0.5
    mask = 1 - inv_mask

    if on_logits:
        predictions = np.log(np.clip(predictions, a_min=1e-8, a_max=1.0)) - np.log(np.clip(1 - predictions, a_min=1e-8, a_max=1.0)) # Logits
    
    weights_per_pred = (weights[:,None] * mask).sum(axis=0)
    predictions = (weights[:,None] * predictions * mask).sum(axis=0) / np.clip(weights_per_pred, a_min=1e-4, a_max=1e5)
    predictions[np.where(weights_per_pred == 0.0)] = 0.5

    if on_logits:
        predictions = 1.0 / (1.0 + np.exp(-predictions)) # Sigmoid
    
    return predictions


def brute_force_finder(eval_func, num_weights, weight_range, max_weights=1e5):
    if (np.log(len(weight_range)) * num_weights) < np.log(2e7):
        weight_tuples = [w for w in product(weight_range, repeat=num_weights)] 
        if len(weight_tuples) > max_weights:
            logger.info("[Weight search] For a full test, we would need %i weight tuples, but limit it to %i" % (len(weight_tuples), max_weights))
            random.seed(42)
            shuffle(weight_tuples)
            weight_tuples = weight_tuples[:max_weights]
    else:
        np.random.seed(42)
        rand_idx = np.random.randint(0, len(weight_range), size=(max_weights, num_weights))
        logger.info("[Weight search] Sampling %i random weight combinations to find best ensemble" % max_weights)
        weight_tuples = [[weight_range[rand_idx[m,n]] for n in range(num_weights)] for m in range(max_weights)]

    best_score = -1
    best_config = None
    for weights in weight_tuples:
        for on_logits in [True, False]:
            score, = eval_func(weights, on_logits=on_logits)
            if score > best_score:
                best_score = score
                best_config = {"weights": weights, "on_logits": on_logits}
                logger.info("[Weight search] New best score of %4.2f%% with config %s" % (best_score*100.0, str(best_config)))
    return best_score, best_config


def mutation(toolbox, ind1, min_weight, max_weight):
    ind2 = toolbox.clone(ind1)
    if random.random() < 0.2:
        scale_factor = random.uniform(a=0.5, b=2.0) # Biased towards positive.
        for i in range(len(ind2)):
            ind2[i] = (ind2[i] - 1) * scale_factor + 1
    else:
        sigma = random.uniform(a=0.02, b=0.2)
        ind2, = tools.mutGaussian(ind2, mu=0.0, sigma=sigma, indpb=0.8)
    for i in range(len(ind2)):
        ind2[i] = min(max(ind2[i], min_weight), max_weight)
        if ind2[i] < 0.2 and random.random() < 0.5:
            ind2[i] = 0.0
    del ind2.fitness.values
    return ind2,


def ind_init(icls, individual_scores, min_weight=0.0, max_weight=4.0):
    if random.random() > 0.5: # Random init
        ind = [random.gauss(1.0, 0.3) for _ in individual_scores]
    else:
        min_scores, max_scores = min(individual_scores), max(individual_scores)
        individual_scores = [(e - min_scores + 0.01) / (max_scores - min_scores) for e in individual_scores]
        sum_scores = sum(individual_scores)
        ind = [random.gauss(e / sum_scores * len(individual_scores), 0.3) for e in individual_scores]
    ind = [min(max(min_weight, w), max_weight) for w in ind]
    return icls(ind)


def EA_ensemble_finder(eval_func, num_weights, individual_scores, population_size=512, min_weight=0.0, max_weight=4.0, num_generations=100):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("individual", ind_init, creator.Individual, individual_scores, min_weight, max_weight)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_func)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", lambda ind: mutation(toolbox, ind, min_weight, max_weight))
    toolbox.register("select", tools.selTournament, tournsize=3)

    hof = tools.HallOfFame(1)
    population = toolbox.population(n=population_size)
    hof.update(population)
    best_score = -1
    best_gen = 0
    for gen in range(num_generations):
        parents = toolbox.select(population, k=len(population))
        offspring = algorithms.varAnd(parents, toolbox, cxpb=0.5, mutpb=0.9)
        fits = toolbox.map(toolbox.evaluate, offspring)
        fits = list(fits)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit 
        population = toolbox.select([i for pop in [population, offspring] for i in pop], k=len(population))
        hof.update(population)
        if hof[0].fitness.values[0] > best_score:
            best_score = hof[0].fitness.values[0]
            best_gen = gen
        elif (gen - best_gen) >= 50:
            logger.info("[EA search] Reinitialize population")
            population = toolbox.population(n=population_size) # Re-init
            best_gen = gen
        if (gen+1) % 20 == 0:
            logger.info("[EA search] Finished %i generations, current max score: %4.2f%%" % (gen+1, hof[0].fitness.values[0]*100.0))
    best_config = {"weights": hof[0], "on_logits": True}
    best_score = hof[0].fitness.values[0]
    return best_score, best_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--regex_dev', type=str,
                        help='Regex expression for dev csv files')
    parser.add_argument('--regex_test', type=str, nargs='+',
                        help='Regex expressions for test csv files')
    args = parser.parse_args()

    dev_files = sorted(glob(args.regex_dev))
    test_files = [sorted(glob(t)) for t in args.regex_test]
    find_ensemble(dev_files, test_files, max_weights=int(1e4))