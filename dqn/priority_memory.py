"""
Class to efficiently store and sample memories based on prority.
Encourages the sampling of influential experiences (those with large 
deviations between expected Q and target values).

Based on:
https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
"""
import random

from sum_tree import SumTree

class PriorityMemory:   

    def __init__(self, capacity):
        """
        Instantiate a priority based memory with capable of holding
        capacity experiences. Memories are sampled with frequency
        based on their priority.
        """
        # Circular buffer array based tree with priorities as node values.
        self.tree = SumTree(capacity)
        self.e = 0.01 # Small constant to ensure all priorities > 0
        self.a = 0.6  # Constant to control the weight of error on priority

    def _getPriority(self, error):
        """
        Convert error to a priority based on the constants "e" and "a"
        """
        return (error + self.e) ** self.a

    def add(self, experience, error):
        """
        Add an experience to memory
        """
        p = self._getPriority(error)
        self.tree.add(p, experience) 

    def sample(self, n):
        """
        Sample n experiences from memory. Experiences selection
        frequency is based on priority.

        Returns:
            - mini_batch: Sequence containing the experiences.
            - indicies: The index of the node associated with each experience 
              so that its priority can be updated.
        """
        mini_batch = []
        indicies = []

        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, _, experience) = self.tree.get(s)
            mini_batch.append(experience)
            indicies.append(idx)

        return mini_batch, indicies

    def update(self, idx, error):
        """
        Update the priority associated with a memory.
        """
        p = self._getPriority(error)
        self.tree.update(idx, p)
