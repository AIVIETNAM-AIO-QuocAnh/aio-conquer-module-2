from src.pipelines.build_pipeline import BuildPipeline
from src.config import load_config

cfg = load_config()
# Thí nghiệm 1: Scaling 
# 2 case: no scale & scale -> model -> result được trả về trong chính biến experiment

    # Case 1 : 
# first_experiment = BuildPipeline(flag_scale = False)
# results = first_experiment.run() 
# first_experment.log(results)

    # Case 2:
# first_experiment = BuildPipeline(flag_scale = True)
# first_experiment.run()
# first_experiment.log('Scaling.csv')

# Thí nghiệm 2: PCA

    # Case 0: được tái sử dụng từ kết quả sau cùng của thí nghiệm 1 (để n_components = None)
second_experiment = BuildPipeline(flag_scale = False, flag_pca = True)
second_experiment.run() 
second_experiment.log('PCA.csv') 
    # Case 1: Scale -> PCA(90%) -> model -> result
second_experiment = BuildPipeline(flag_scale = True, flag_pca = True, n_components = cfg['explain_var']['A'])
second_experiment.run() 
second_experiment.log('PCA.csv') 

    # Case 2: Scale -> PCA(90%) -> model -> result
# second_experiment = BuildPipeline(flag_scale = True, flag_pca = True, n_components = cfg['explain_var']['B'])
# second_experiment.run() 
# second_experiment.log('PCA.csv') 