task surrogateScores : unli
  < snli=$out@snliQRels
  < uSnli=$out@uSnliQRels
  > out="surrogate.scores"
  :: aggregator=@
{
  python $unli/scripts/compute_surrogates.py \
    --snli $snli \
    --usnli $uSnli/train.qrels \
    --aggregator $aggregator \
    > $out
}

task snliWithSurrogatesDataset : unli
  < snli=$out@snliQRels
  < uSnli=$out@uSnliQRels
  < surrogate=$out@surrogateScores
  > out
{
  mkdir -p $out
  python $unli/scripts/gen_snli_with_surrogates_dataset.py \
    --input $snli/train.qrels \
    --surrogate $surrogate \
    > $out/train.qrels
  ln -s $snli/train.l $out/train.l
  ln -s $snli/train.r $out/train.r
  ln -s $snli/dev.l $out/dev.l
  ln -s $snli/dev.r $out/dev.r
  ln -s $snli/test.l $out/test.l
  ln -s $snli/test.r $out/test.r
  ln -s $uSnli/dev.qrels $out/dev.qrels
  ln -s $uSnli/test.qrels $out/test.qrels
}

task snliCombinedWithUSnliDataset : unli
  < snli=$out@snliQRels
  < uSnli=$out@uSnliQRels
  < surrogate=$out@surrogateScores
  > out
{
  mkdir -p $out
  python $unli/scripts/gen_combined_fallback_surrogate_dataset.py \
    --snli $snli/train.qrels \
    --usnli $uSnli/train.qrels \
    --surrogate $surrogate \
    > $out/train.qrels
  ln -s $snli/train.l $out/train.l
  ln -s $snli/train.r $out/train.r
  ln -s $snli/dev.l $out/dev.l
  ln -s $snli/dev.r $out/dev.r
  ln -s $snli/test.l $out/test.l
  ln -s $snli/test.r $out/test.r
  ln -s $uSnli/dev.qrels $out/dev.qrels
  ln -s $uSnli/test.qrels $out/test.qrels
}

task uSnliDataset
  < snli=$out@snliQRels
  < uSnli=$out@uSnliQRels
  > out
{
  mkdir -p $out
  ln -s $snli/train.l $out/train.l
  ln -s $snli/train.r $out/train.r
  ln -s $snli/dev.l $out/dev.l
  ln -s $snli/dev.r $out/dev.r
  ln -s $snli/test.l $out/test.l
  ln -s $snli/test.r $out/test.r
  ln -s $uSnli/train.qrels $out/train.qrels
  ln -s $uSnli/dev.qrels $out/dev.qrels
  ln -s $uSnli/test.qrels $out/test.qrels
}

plan Data {
  reach snliWithSurrogatesDataset via (Aggregator: mean)
  reach snliCombinedWithUSnliDataset via (Aggregator: mean)
  reach uSnliDataset
}
