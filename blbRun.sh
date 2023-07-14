# 在服务器以yarn模式运行一个gcForest，数据集为uci_adult，更新之后，class不再是GCForestSequence，而是GCForestAdult，Sequence用来做多个数据集的统一入口，以后运行需要制定数据集，如下
# --dataset uci_adult 或 covertype, watch_acc, susy, higgs
spark-submit --master yarn \
  --class org.apache.spark.ml.examples.BLBGCForestSequence \
  --executor-cores 2 \
  --num-executors 16 \
  --conf spark.dynamicAllocation.minExecutors=16 \
  --conf spark.dynamicAllocation.maxExecutors=16 \
  gcforest-1.0-SNAPSHOT-jar-with-dependencies.jar \
  --dataset uci_adult \
  --train linyigang/data/uci_adult/adult.data \
  --test linyigang/data/uci_adult/adult.test \
  --features linyigang/data/uci_adult/features \
  --classNum 2 \
  --casTreeNum 5 \
  --rfNum 1 \
  --crfNum 1 \
  --subRFNum 3 \
  --maxIteration 2 \
  --lambda 0.8

# 在服务器以yarn模式运行1个blb-gcForest，数据集为Covertype;
# 需要设置内存，否则OOM；样本多时需要让executor-memory增大，否则无法训练树；树多时要让driver-memory增大，否则无法收集森林。
spark-submit --master yarn \
  --class org.apache.spark.ml.examples.BLBGCForestSequence \
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
  --classNum 7 \
  --casTreeNum 5 \
  --rfNum 1 \
  --crfNum 1 \
  --subRFNum 3 \
  --maxIteration 2 \
  --lambda 0.6

# 在服务器以yarn模式运行1个gcForest，数据集为watch_acc;
# 需要设置内存，否则OOM；样本多时需要让executor-memory增大，否则无法训练树；树多时要让driver-memory增大，否则无法收集森林。
spark-submit --master yarn \
  --class org.apache.spark.ml.examples.BLBGCForestSequence \
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
  --classNum 18 \
  --casTreeNum 5 \
  --rfNum 1 \
  --crfNum 1 \
  --subRFNum 3 \
  --maxIteration 2 \
  --lambda 0.6

# 在服务器以yarn模式运行1个gcForest，数据集为 SUSY;
# 这个数据集已经无法在我自己的电脑跑起来了，OOM
# 需要设置内存，否则OOM；样本多时需要让executor-memory增大，否则无法训练树；树多时要让driver-memory增大，否则无法收集森林。
spark-submit --master yarn \
  --class org.apache.spark.ml.examples.BLBGCForestSequence \
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
  --classNum 2 \
  --casTreeNum 2 \
  --rfNum 1 \
  --crfNum 1 \
  --subRFNum 3 \
  --maxIteration 2 \
  --lambda 0.6

# 在服务器以yarn模式运行1个gcForest，数据集为 HIGGS;
# 这个数据集已经无法在我自己的电脑跑起来了，OOM
# 需要设置内存，否则OOM；样本多时需要让executor-memory增大，否则无法训练树；树多时要让driver-memory增大，否则无法收集森林。
spark-submit --master yarn \
  --class org.apache.spark.ml.examples.BLBGCForestSequence \
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
  --classNum 2 \
  --casTreeNum 2 \
  --rfNum 1 \
  --crfNum 1 \
  --subRFNum 3 \
  --maxIteration 2 \
  --lambda 0.6
