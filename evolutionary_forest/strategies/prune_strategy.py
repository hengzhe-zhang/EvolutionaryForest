class PruneStrategy(dict):
    def __init__(self, *arg, **kw):
        self.semantic_prune_and_plant = kw.get('semantic_prune_and_plant', False)
        self.double_tournament = kw.get('double_tournament', False)
        super(PruneStrategy, self).__init__(*arg, **kw)
