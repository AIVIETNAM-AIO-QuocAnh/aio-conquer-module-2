from src.pipelines.build_pipeline import BuildPipeline

# Thí nghiệm 1: Scaling 
# 2 case: no scale & scale -> model -> result được trả về trong chính biến experiment

# Case 1 : 
# first_experiment = BuildPipeline(flag_scale = False)
# results = first_experiment.run() 
# first_experment.log(results)

# Case 2:
first_experiment = BuildPipeline(flag_scale = True)
first_experiment.run()
# first_experiment.log()

# Thí nghiệm 2: PCA

# Case 0: được tái sử dụng từ kết quả sau cùng của thí nghiệm 1 
 
# Case 1: Scale -> PCA(90%) -> model -> result
# second_experiment = BuildPipeline(flag_scale = True, flag_pca = True, n_components = 0.7)
# second_experiment.run() 
# second_experment.log()

# Case 2: Scale -> PCA(90%) -> model -> result
# second_experiment = BuildPipeline(flag_scale = True, flag_pca = True, n_components = 0.9)
# second_experiment.run() 
# second_experment.log()