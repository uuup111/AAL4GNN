# config for stages, waiting for adding

OGBN_ARXIV = {
    'input': ['origin', 'in-domain', 'out-domain'],
    'transform': ['none', 'masked'],
    'representation': ['none', 'sgTransformer', 'transformer'],
    'output': ['link_prediction', 'community_detection']
}