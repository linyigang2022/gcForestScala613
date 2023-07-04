# 在服务器以yarn模式运行一个gcForest，数据集为uci_adult，更新之后，class不再是GCForestSequence，而是GCForestAdult，Sequence用来做多个数据集的统一入口，以后运行需要制定数据集，如下
# --dataset uci_adult 或 covertype, watch_acc, susy, higgs
spark-submit --master yarn \
  --class  org.apache.spark.ml.examples.UCI_adult.GCForestSequence \
  --executor-cores 2 \
  --num-executors 16 \
  --conf spark.dynamicAllocation.minExecutors=16 \
  --conf spark.dynamicAllocation.maxExecutors=16 \
  gcforest-1.0-SNAPSHOT-jar-with-dependencies.jar \
  --dataset uci_adult \
  --train linyigang/data/uci_adult/adult.data \
  --test linyigang/data/uci_adult/adult.test \
  --features linyigang/data/uci_adult/features \
  --casTreeNum 5 \
  --rfNum 1 \
  --crfNum 1

# 在服务器以yarn模式运行4颗并行的gcForest，仅运行的类名需要改
spark-submit --master yarn \
  --class org.apache.spark.ml.tmp.ParallelGCForest\
  --executor-cores 2 \
  --num-executors 16 \
  --conf spark.dynamicAllocation.minExecutors=16 \
  --conf spark.dynamicAllocation.maxExecutors=16 \
  gcforest-1.0-SNAPSHOT-jar-with-dependencies.jar \
  --dataset uci_adult \
  --train linyigang/data/uci_adult/adult.data \
  --test linyigang/data/uci_adult/adult.test \
  --features linyigang/data/uci_adult/features \
  --casTreeNum 5 \
  --rfNum 1 \
  --crfNum 1

# 在服务器以yarn模式运行1个gcForest，数据集为Covertype;
# 需要设置内存，否则OOM；样本多时需要让executor-memory增大，否则无法训练树；树多时要让driver-memory增大，否则无法收集森林。
# 16*2 casTreeNum=100 一次rf训练（3叉验证的一次）需 605s 581s 571s，准确率 87.512%、87.147%、87.288%;第一次完成的三叉验证训练准确率 87.316%，测试准确率 87.573%；
#                     一次crf训练需 172s 176s 179s，准确率 69.833%、68.818%、68.234%；第一次完成的三叉验证训练准确率 68.962%，测试准确率 69.276%；
#                     第一层 耗时 2658.088s 训练准确率 78.139%，测试准确率 78.424%
spark-submit --master yarn \
  --class org.apache.spark.ml.examples.Covertype.GCForestCovertype \
  --executor-cores 2 \
  --num-executors 16 \
  --driver-memory 20G \
  --executor-memory 20G \
  --conf spark.dynamicAllocation.minExecutors=16 \
  --conf spark.dynamicAllocation.maxExecutors=16 \
  gcforest-1.0-SNAPSHOT-jar-with-dependencies.jar \
  --dataset covertype \
  --train linyigang/data/covertype/covtype.data \
  --features linyigang/data/covertype/features \
  --casTreeNum 5 \
  --rfNum 1 \
  --crfNum 1

# 在服务器以yarn模式运行1个gcForest，数据集为watch_acc;
# 需要设置内存，否则OOM；样本多时需要让executor-memory增大，否则无法训练树；树多时要让driver-memory增大，否则无法收集森林。
# 16*2 casTreeNum=5 一次rf训练（3叉验证的一次）需10mins，准确率 91.084%
# 16*2 casTreeNum=2 一次rf训练（3叉验证的一次）需 188s 182s 183s，准确率 90.565%、90.256%、90.145%；第一次完成的三叉验证训练准确率 90.322%，测试准确率 93.384%；
#                    一次crf训练需 41s 46s 41s，准确率 36.571%、45.794%、44.217%；第一次完成的三叉验证训练准确率 42.199%，测试准确率 58.854%；
#                    第一层 耗时 759.12s 训练准确率 66.260%，测试准确率 76.119%
spark-submit --master yarn \
  --class org.apache.spark.ml.examples.Watch_acc.GCForestWatchAcc \
  --executor-cores 2 \
  --num-executors 16 \
  --driver-memory 10G \
  --executor-memory 20G \
  --conf spark.dynamicAllocation.minExecutors=16 \
  --conf spark.dynamicAllocation.maxExecutors=16 \
  gcforest-1.0-SNAPSHOT-jar-with-dependencies.jar \
  --dataset watch_acc \
  --train linyigang/data/watch_acc/watch_acc.data \
  --features linyigang/data/watch_acc/features \
  --casTreeNum 5 \
  --rfNum 1 \
  --crfNum 1

# 在服务器以yarn模式运行1个gcForest，数据集为 SUSY;
# 这个数据集已经无法在我自己的电脑跑起来了，OOM
# 需要设置内存，否则OOM；样本多时需要让executor-memory增大，否则无法训练树；树多时要让driver-memory增大，否则无法收集森林。
# 16*2 casTreeNum=2 一次rf训练（3叉验证的一次）196s 185s 166s，准确率 73.156% 73.275% 73.317%，一次完整三叉验证 训练准确率 73.249%，测试准确率 78.062%
#                   一次crf训练（3叉验证的一次）131s 101s 166s，准确率 78.413% 78.339% 73.317%，一次完整三叉验证 训练准确率 78.306%，测试准确率 79.226%
#                   第一层 耗时 1066.011s 训练准确率 75.864%，测试准确率 78.680%
spark-submit --master yarn \
  --class org.apache.spark.ml.examples.SUSY.GCForestSUSY \
  --executor-cores 2 \
  --num-executors 16 \
  --driver-memory 10G \
  --executor-memory 20G \
  --conf spark.dynamicAllocation.minExecutors=16 \
  --conf spark.dynamicAllocation.maxExecutors=16 \
  gcforest-1.0-SNAPSHOT-jar-with-dependencies.jar \
  --train linyigang/data/susy/SUSY.data \
  --dataset susy \
  --features linyigang/data/susy/features \
  --casTreeNum 2 \
  --rfNum 1 \
  --crfNum 1

# 在服务器以yarn模式运行1个gcForest，数据集为 HIGGS;
# 这个数据集已经无法在我自己的电脑跑起来了，OOM
# 需要设置内存，否则OOM；样本多时需要让executor-memory增大，否则无法训练树；树多时要让driver-memory增大，否则无法收集森林。
# 16*2 casTreeNum=2 一次rf训练（3叉验证的一次）436s 381s 381s，准确率 64.873% 64.842% 67.712%，一次完整的三叉验证 训练准确率 64.809%，测试准确率 70.845%
#                   一次rf训练（3叉验证的一次）236s 188s 206s，准确率 66.326% 65.126% 65.495%，一次完整的三叉验证 训练准确率 65.649%，测试准确率 68.611%
#                   第一层耗时 2415.189s，训练准确率 65.229%，测试准确率 69.728%
spark-submit --master yarn \
  --class org.apache.spark.ml.examples.HIGGS.GCForestHIGGS \
  --executor-cores 2 \
  --num-executors 16 \
  --driver-memory 10G \
  --executor-memory 20G \
  --conf spark.dynamicAllocation.minExecutors=16 \
  --conf spark.dynamicAllocation.maxExecutors=16 \
  gcforest-1.0-SNAPSHOT-jar-with-dependencies.jar \
  --dataset higgs \
  --train linyigang/data/higgs/HIGGS.csv \
  --features linyigang/data/higgs/features \
  --casTreeNum 2 \
  --rfNum 1 \
  --crfNum 1