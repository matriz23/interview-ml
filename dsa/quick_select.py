import random


def quick_select(nums: list, left: int, right: int, k: int):
    if left > right:
        return None
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
    if i == len(nums) - k:
        return pivot
    elif i > len(nums) - k:
        return quick_select(nums, left, i - 1, k)
    else:
        return quick_select(nums, i + 1, right, k)


if __name__ == "__main__":
    nums = [3, 6, 8, 10, 1, 2, 1]
    print(quick_select(nums, 0, len(nums) - 1, 4))
    print(quick_select(nums, 0, len(nums) - 1, 10))
