import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

_deepcopy_dispatch = d = {}
def _deepcopy_atomic(x, memo):
    return x
def _keep_alive(x, memo):
    """Keeps a reference to the object x in the memo.

    Because we remember objects by their id, we have
    to assure that possibly temporary objects are kept
    alive by referencing them.
    We store a reference at the id of the memo, which should
    normally not be used unless someone tries to deepcopy
    the memo itself...
    """
    try:
        memo[id(memo)].append(x)
    except KeyError:
        # aha, this is the first one :-)
        memo[id(memo)]=[x]
def deepcopy(x, memo=None, _nil=[]):
    """Deep copy operation on arbitrary Python objects.

    See the module's __doc__ string for more info.
    """

    if memo is None:
        memo = {}

    d = id(x)
    y = memo.get(d, _nil)
    if y is not _nil:
        return y

    cls = type(x)

    copier = _deepcopy_dispatch.get(cls)
    if copier is not None:
        y = copier(x, memo)
    else:
        if issubclass(cls, type):
            y = _deepcopy_atomic(x, memo)
        else:
            copier = getattr(x, "__deepcopy__", None)
            if copier is not None:
                y = copier(memo)
            else:
                reductor = dispatch_table.get(cls)
                if reductor:
                    rv = reductor(x)
                else:
                    reductor = getattr(x, "__reduce_ex__", None)
                    if reductor is not None:
                        rv = reductor(4)
                    else:
                        reductor = getattr(x, "__reduce__", None)
                        if reductor:
                            rv = reductor()
                        else:
                            raise Error(
                                "un(deep)copyable object of type %s" % cls)
                if isinstance(rv, str):
                    y = x
                else:
                    y = _reconstruct(x, memo, *rv)

    # If is its own copy, don't memoize.
    if y is not x:
        memo[d] = y
        _keep_alive(x, memo) # Make sure x lives at least as long as d
    return y


 





# txt 
evidence = pd.read_csv('./evidence.txt', sep='\t')
msms_df = pd.read_csv('./msms.txt', sep='\t')
msms = deepcopy(msms_df).rename(columns={'id': 'Best MS/MS'})

peptides_df = pd.merge(evidence, msms[['Best MS/MS', 'Matches', 'Intensities']], on='Best MS/MS', how='left')
peptides_df['CMS'] = peptides_df['Charge'].map(str) + peptides_df['Modified sequence']

# #target_peptides_df = peptides_df[peptides_df['Reverse'].isnull()]
selected_columns = peptides_df[[ 'CMS','Score','Intensities','Sequence','Type','Matches', 'Length','PEP','Leading razor protein',  'Modifications', 'Modified sequence', 'Proteins', 'Charge', 'Reverse']]

print(len(selected_columns))


valid_psm = selected_columns[(selected_columns['PEP'] < 0.01)&
                              (~selected_columns['Matches'].isna())
                               &(~selected_columns['Intensities'].isna())
 
                             & (selected_columns['Score']>20)
                             &(selected_columns['Charge']<=6)
                             & (selected_columns['Length']<=47)
                       
                               ] 




print(len(valid_psm))






