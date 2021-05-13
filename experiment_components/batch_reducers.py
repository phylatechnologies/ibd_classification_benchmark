import batch_reduction.zero_centering as zero_centering
import batch_reduction.log_zero_centering as log_zero_centering
import batch_reduction.mmuphin as mmuphin


def identity(df, ignore, *kwargs):
    return df


batch_reducers = {'BRN': (identity, {}),
                  'BRZC': (zero_centering.zero_center, {'batch_column': 'col_site'}),
                  'BRLZC': (log_zero_centering.log_zero_center, {'batch_column': 'col_site'}),
                  'BRMNC': (mmuphin.remove_all_batch_effects, {"covariates": ['stool_biopsy'],
                                                               "batch_order": {0: 'col_site'}, 'drop_first': False,
                                                               'add_mode': 5, 'eb': True, 'grand_mean': False}),
                  'BRMCC': (mmuphin.remove_all_batch_effects, {"covariates": ['stool_biopsy', 'uc_cd'],
                                                               "batch_order": {0: 'col_site'}, 'drop_first': False,
                                                               'add_mode': 5, 'eb': True, 'grand_mean': False}),
                  'BRMCCL': (mmuphin.remove_all_batch_effects, {"covariates": ['stool_biopsy', 'uc_cd'],
                                                                "batch_order": {0: 'col_site'}, 'drop_first': False,
                                                                'add_mode': 5, 'eb': True, 'grand_mean': False}),
                  'BRMCCS': (mmuphin.remove_all_batch_effects, {"covariates": ['stool_biopsy', 'uc_cd'],
                                                                "batch_order": {0: 'col_site'}, 'drop_first': False,
                                                                'add_mode': 5, 'eb': True, 'grand_mean': False}),
                  'BRMNCS': (mmuphin.remove_all_batch_effects, {"covariates": ['stool_biopsy'],
                                                                "batch_order": {0: 'col_site'}, 'drop_first': False,
                                                                'add_mode': 5, 'eb': True, 'grand_mean': False}),
                  'BRMNSS': (mmuphin.remove_all_batch_effects, {"covariates": ['stool_biopsy'],
                                                                "batch_order": {0: 'studyID'}, 'drop_first': False,
                                                                'add_mode': 5, 'eb': True, 'grand_mean': False}),  
                  'BRMCSL': (mmuphin.remove_all_batch_effects, {"covariates": ['stool_biopsy', 'uc_cd'],
                                                                "batch_order": {0: 'studyID'}, 'drop_first': False,
                                                                'add_mode': 5, 'eb': True, 'grand_mean': False})                                                                                                                                                  
                  }

