import heapq


class MinPriorityQueue:
    def __init__(self):
        self.heap = []

    def push(self, item):
        heapq.heappush(self.heap, item)

    def append(self, item):
        self.push(item)

    def pop(self, index):
        assert index == 0
        return heapq.heappop(self.heap)

    def peek(self):
        return self.heap[0] if self.heap else None

    def is_empty(self):
        return len(self.heap) == 0

    def __iter__(self):
        return iter(self.heap)

    def __len__(self):
        return len(self.heap)


if __name__ == "__main__":
    pq = MinPriorityQueue()
    pq.push(5)
    pq.push(1)
    pq.push(3)

    print("Smallest element:", pq.pop())  # Outputs 1
    print("Next smallest element:", pq.pop())  # Outputs 3
