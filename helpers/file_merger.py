import pandas as pd
import os

import pandas as pd
import os
from glob import glob
def merge_results(results_dir):
	main_df = pd.DataFrame()
	for file in glob(results_dir):
		df = pd.read_csv(file, index_col=0)
		main_df = pd.concat([main_df,df])
	return main_df
