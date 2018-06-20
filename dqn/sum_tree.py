import numpy

class SumTree:
    """
    Tree used to store prioritized experience replay memories efficently.
    
    Implementation based on:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    """
    

    def __init__(self, capacity):
        """
        The tree is implemented as an array, where a node's parent is always 
        located at (child_idx - 1) // 2. The tree requires 2*n - 1 nodes to
        store n experiences. 
        """
        self.capacity = capacity
        self.tree = numpy.zeros( 2*capacity - 1 )
        self.data = numpy.zeros(capacity, dtype=object)

        # Write index to overwrite old values - makes the tree a circular buffer.
        self.write = 0 

    def _propagate(self, idx, change):
        """
        Each memory is stored with a priority representing the error (e) between
        the predicted Q value and target value for the given state. When the
        memory is inserted into the tree, its error is propagated through
        to the root such that the leaf encompasses a range of priorities equal to e.

        To modify the prority associated with a leaf, the difference between the old
        and new prority (change) must be propagated through to the root.

        Arguments:
            - idx: Index of the node to update.
            - change: Amount the node's priority is changed by.
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            # Propagate the change recursively so the path from
            # the updated node to the root reflects the new priority.
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """
        Search the tree recursively for the index of the leaf with a 
        priority region whic includes s.

        Ultimately, batch_size memories will be sampled from the tree.
        This sampling breaks the total error sum into batch_size regions.
        A memory is then sampled from each region, thus a larger priority
        makes a memory more likely to be selected from its region, or even
        lets the memory span multiple regions.

        Arguments: 
            - idx: The index of the current node in the search
            - s: The remaining priority range.

        Returns:
            The index of the leaf with a priority range which includes s.
        """
        # Determine the parent's left and right nodes
        left_child = 2 * idx + 1
        right_child = left_child + 1

        # Leaves do not have children, thus the indices of their
        # children will be invalid.
        if left_child >= len(self.tree):
            return idx

        # The tree encodes memories into ranges with values ranging
        # from 0 to the value of the root. Larger start values
        # represent leaves located at larger indices (futher to the right
        # in the tree). An interior node's value represents the range of 
        # values its children encompass - equivalently, this is range of values 
        # in its segment of the tree.
        if s <= self.tree[left_child]:
            return self._retrieve(left_child, s)
        else:
            # Remove the lower poriton of the priority associated 
            # with the left child.
            return self._retrieve(right_child, s-self.tree[right_child])

    def total(self):
        """
        The value of the root encodes the sum of all priorities.
        """
        return self.tree[0]

    def add(self, p, data):
        """
        Overwrite the oldest memory - tracked with the "write" field

        Arguments:
            - p: The priority of the experience
            - date: The experience.
        """
        # Leaf nodes start at index self.capacity - 1
        idx = self.write + self.capacity - 1

        # Write the new data and set the new priority.
        self.data[self.write] = data
        self.update(idx, p)
        
        # Circular buffer - replace oldest memories.
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        """
        Update the prority of a leaf node. This requires propagating
        the change to the root so the range of prorities encompassed
        by the leaf is updated.
        
        Arguments:
            - idx: The index within the array to update
            - p: The new priority of the node at index idx.
        """
        # The interior nodes must be updated to reflect the change
        # in prority, as these nodes values are the priorities
        # from various subsections of the tree.
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """
        Recursively search the tree to determine the leaf assigned
        to the priority range which includes s.

        Returns:
            - The index of the leaf.
            - The priority of the leaf.
            - The memory associated with the leaf.
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])