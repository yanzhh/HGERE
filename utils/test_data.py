
import random
from data import _create_shuffled_batches

n_ents_by_sent = [random.randint(0, 32) for _ in range(1000)]



reordered = []
for thing in _create_shuffled_batches(n_ents_by_sent, 32):
    reordered.extend(thing)
assert len(n_ents_by_sent) == len(reordered)

"""        
for i in range(20):
    sent_id = size_iter.next_sent_id("smallest")
    if sent_id is None:
        print("None")
        continue
    size = n_ents_by_sent[sent_id]
    print(len(size_iter.size_to_sent_ids))
    print(sent_id, size)
    #print(n_ents_by_sent)
    """
