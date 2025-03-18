import random


def quick_sort(nums: list, left: int, right: int):
    if left >= right:
        return
    p_idx = random.randint(left, right)
    nums[left], nums[p_idx] = nums[p_idx], nums[left]
    pivot = nums[left]
    i, j = left, right
    while i < j:
        while i < j and nums[j] >= pivot:
            j -= 1
        nums[i] = nums[j]
        while i < j and nums[i] <= pivot:
            i += 1
        nums[j] = nums[i]
    nums[i] = pivot
    quick_sort(nums, left, i - 1)
    quick_sort(nums, i + 1, right)


if __name__ == "__main__":
    nums = [3, 6, 8, 10, 1, 2, 1]
    quick_sort(nums, 0, len(nums) - 1)
    print(nums)
