from auxiliary_tasks.clustering import ClusteringTask


def add_auxiliary_task(config):
    aux_tasks = []
    task = None
    for task_cfg in config['auxiliary_tasks']:
        if task_cfg['type'] == 'clustering':
            task = add_clustering_task(task_cfg)
        aux_tasks.append(task)
    return aux_tasks


def add_clustering_task(task_cfg):
    task = ClusteringTask(task_cfg)
    return task

