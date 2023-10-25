from deap.gp import Primitive, Terminal


class IntronPrimitive(Primitive):
    __slots__ = (
        "name",
        "arity",
        "args",
        "ret",
        "seq",
        "corr",
        "level",
        "equal_subtree",
        "hash_id",
    )

    @property
    def intron(self):
        return self.corr < 0.01

    def __init__(self, name, args, ret):
        super().__init__(name, args, ret)
        self.corr = 0
        self.level = 0
        self.equal_subtree = -1
        self.hash_id = 0


class IntronTerminal(Terminal):
    __slots__ = ("name", "value", "ret", "conv_fct", "corr", "level", "hash_id")

    @property
    def intron(self):
        return self.corr < 0.01

    def __init__(self, terminal, symbolic, ret):
        super().__init__(terminal, symbolic, ret)
        self.corr = 0
        self.level = 0
        self.hash_id = 0
