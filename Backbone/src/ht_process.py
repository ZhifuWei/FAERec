import math
from collections import Counter
from typing import List, Set, Dict

# split head/tail items
def classify_head_and_tail(user_seq: List[List[int]],
                           head_ratio: float,
                           head_items: Set[int],
                           tail_items: Set[int],
                           head_users: Set[int],
                           tail_users: Set[int]) -> tuple:
    item_cnt = Counter([i for seq in user_seq for i in seq])
    sorted_items = sorted(item_cnt, key=item_cnt.get, reverse=True)
    split_item = int(len(sorted_items) * head_ratio)
    
    item_threshold = item_cnt[sorted_items[split_item - 1]] if split_item > 0 else 0
    
    min_item_interaction = item_cnt[sorted_items[-1]] if sorted_items else 0
    
    head_items.clear()
    head_items.update(sorted_items[:split_item])
    tail_items.clear()
    tail_items.update(sorted_items[split_item:])

    intervals = [
        (1, 4, "1-4"),
        (5, 9, "5-9"),
        (10, 19, "10-19"),
        (20, 49, "20-49"),
        (50, float('inf'), "50+")
    ]
    interval_counts = {name: 0 for _, _, name in intervals}
    for count in item_cnt.values():
        for low, high, name in intervals:
            if low <= count <= high:
                interval_counts[name] += 1
                break

    user_cnt = {uid: len(seq) for uid, seq in enumerate(user_seq)}
    sorted_users = sorted(user_cnt, key=user_cnt.get, reverse=True)
    split_user = int(len(sorted_users) * head_ratio)
    
    user_threshold = user_cnt[sorted_users[split_user - 1]] if split_user > 0 else 0
    
    min_user_seq_length = user_cnt[sorted_users[-1]] if sorted_users else 0
    
    head_users.clear()
    head_users.update(sorted_users[:split_user])
    tail_users.clear()
    tail_users.update(sorted_users[split_user:])

    print(f"Item Classification Threshold: Minimum occurrence count for head items is {item_threshold}")
    print(f"Minimum Item Interaction Count:{min_item_interaction}")
    print(f"Minimum User Sequence Length:{min_user_seq_length}")
    print(f"Number of head items:{len(head_items)},Number of items in the tail:{len(tail_items)}")
    print("\nItem Occurrence Frequency Range Distribution:")
    for name, cnt in interval_counts.items():
        print(f"{name} Number of items included:{cnt}")

    avg_len = int(math.floor(sum(len(seq) for seq in user_seq) / len(user_seq)))
    
    return avg_len, user_cnt, dict(item_cnt)