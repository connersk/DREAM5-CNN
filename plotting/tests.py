'''
Define tests
'''
import copy

defaults = dict(
    min_epochs=3,
    max_epochs=25,
    batchsize=100,
    learning_rate=0.001,
    weight_decay=0.001,
    l1_reg=True,
    no_pad=False,
    motif_detectors=16,
    motif_len=24,
    fc_nodes=16,
    dropout=0.5
    )


TESTS = {}

TESTS[0] = copy.deepcopy(defaults)

# 1 layer tests

MOTIF_DETECTORS= [16,32,64,128]
MOTIF_LEN = [4,8,12,24]
FC_NODES = [16,32,64]

n = 1
for md in MOTIF_DETECTORS:
    for ml in MOTIF_LEN:
        for fc in FC_NODES:
            test = copy.deepcopy(defaults)
            test.update(dict(
                motif_detectors=md,
                motif_len=ml,
                fc_nodes=fc
                ))
            TESTS[n] = test
            n += 1

DROPOUT = [0,.25,.5]
W_DECAY = [0,.0001,.001,.01,.1]

for wd in W_DECAY:
    for drop in DROPOUT:
        test = copy.deepcopy(defaults)
        test.update(dict(
                dropout=drop,
                weight_decay=wd
        ))
        TESTS[n] = test
        n += 1
