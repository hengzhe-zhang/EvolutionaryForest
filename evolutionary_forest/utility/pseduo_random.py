import random


class PseudoRandom:
    def __init__(self, length):
        """
        Initialize the pseudo-random generator.

        Args:
            length (int): The range size for the pseudo-random indices (0 to length-1).
        """
        self.length = length
        self.permutation = list(range(length))
        self.index = 0
        self.shuffle()

    def shuffle(self):
        """
        Shuffle the permutation to ensure a new random sequence.
        """
        self.index = 0
        random.shuffle(self.permutation)

    def randint(self):
        """
        Return the next pseudo-random index, ensuring all indices appear exactly once
        every `self.length` calls.

        Returns:
            int: A pseudo-random index.
        """
        value = self.permutation[self.index]
        self.index += 1
        if self.index >= self.length:
            self.shuffle()
        return value


if __name__ == "__main__":
    gene_length = 5
    pseudo_random = PseudoRandom(gene_length)

    # Generate indices
    for _ in range(15):  # Three cycles through the gene
        print(pseudo_random.randint(), end=" ")
