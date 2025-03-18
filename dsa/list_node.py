class ListNode:
    def __init__(self, val=-1, next=None):
        self.val = val
        self.next = next


def build_node(arr):
    dummy = ListNode()
    cur = dummy
    for x in arr:
        cur.next = ListNode(val=x)
        cur = cur.next
    return dummy.next


def print_node(node):
    while node:
        print(node.val, end=" -> " if node.next else "\n")
        node = node.next


def middle_node(head, split=False):
    slow = fast = head
    pre = None
    while fast and fast.next:
        pre = slow
        slow = slow.next
        fast = fast.next.next
    if split:
        pre.next = None
    return slow


def reverse_node(head):
    pre = None
    cur = head
    while cur:
        nxt = cur.next
        cur.next = pre
        pre = cur
        cur = nxt
    return pre
