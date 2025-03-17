"""
MinHeap: A class that implements a minimum heap.
"""


class MinHeap:
    def __init__(self):
        self.heap = []

    def parent(self, idx):
        return (idx - 1) // 2

    def left_child(self, idx):
        return idx * 2 + 1

    def right_child(self, idx):
        return idx * 2 + 2

    def heappush(self, x):
        self.heap.append(x)
        self._heap_up(len(self.heap) - 1)

    def heappop(self):
        if len(self.heap) == 0:
            raise IndexError("Heap is empty")
        self._swap(0, len(self.heap) - 1)
        min_val = self.heap.pop()
        self._heap_down(0)
        return min_val

    def get_min(self):
        return self.heap[0]

    def _swap(self, i, j):
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def _heap_up(self, i):
        while i > 0 and self.heap[self.parent(i)] > self.heap[i]:
            self._swap(i, self.parent(i))

    def _heap_down(self, i):
        idx = i
        left = self.left_child(i)
        if left < len(self.heap) and self.heap[left] < self.heap[idx]:
            idx = left

        right = self.right_child(i)
        if right < len(self.heap) and self.heap[right] < self.heap[idx]:
            idx = right

        if idx != i:
            self._swap(idx, i)
            self._heap_down(idx)


if __name__ == "__main__":
    heap = MinHeap()
    heap.heappush(3)
    heap.heappush(1)
    print(heap.heappop())
    print(heap.get_min())
    heap.heappush(4)
    print(heap.get_min())
    heap.heappush(2)
    print(heap.get_min())
