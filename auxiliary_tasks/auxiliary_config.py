from auxiliary_tasks.clustering import ClusteringTask
from auxiliary_tasks.degree_prediction import DegreePredictionTask


def add_auxiliary_task(config):
    aux_tasks = []
    task = None
    for task_cfg in config['auxiliary_tasks']:
        if task_cfg['type'] == 'clustering':
            task = add_clustering_task(task_cfg)
        elif task_cfg['type'] == 'degree_prediction':
            task = add_degree_prediction_task(task_cfg)
        aux_tasks.append(task)
    return aux_tasks


def add_clustering_task(task_cfg):
    task = ClusteringTask(task_cfg)
    return task


def add_degree_prediction_task(task_cfg):
    task = DegreePredictionTask(task_cfg)
    return task

