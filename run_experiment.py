from pipeline_workflow.pipeline import run_experiments

# *****************************************************
# ******** EXPERIMENT COMPONENTS ********************** 
# *****************************************************
data = ['otu', 'genus','species']
norm = ['TSS']
batch = ['BRMNCS','BRMCCL'] 
models =  ['MLP']
norm2 = ['NOT']

# *****************************************************
# ******** ASSEMBLING COMBOS **************************
# *****************************************************s

exp_list = []
for d in data:
    for n in norm:
        for b in batch:
            for m in models:
                for n2 in norm2:
                    experiment = '-'.join([d, n, b, m, n2])
                    exp_list.append(experiment) 
print('Number of experiments: {}'.format(len(exp_list)))

results_dict = run_experiments(exp_list)

# 'BRN' === 'TSS', 'CLR', 'LAS', 'NOT', 'VST'
# 'BRZC' === 'TSS', 'CLR', 'LAS', 'NOT', 'VST'
# 'BRZLC' === 'TSS', 'LAS', 'NOT'
# 'BRMCC' === 'TSS'
# 'BRMNC' === 'TSS'
# 'BRMCCL' === 'TSS'


 #