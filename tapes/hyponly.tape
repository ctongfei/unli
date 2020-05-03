task hypOnlyDataset : unli
  < dataset=(Mode:
    surrogate=$out@snliWithSurrogatesDataset
    combined=$out@snliCombinedWithUSnliDataset
    fineTune=$out@uSnliDataset
    rawSnli=$out@snliQRels
  )
  < snli=$out@snliQRels
  > out
{
  mkdir -p $out
  python $unli/scripts/remove_premises.py $snli/train.l > $out/train.l
  ln -s $snli/train.r $out/train.r
  ln -s $snli/dev.l $out/dev.l
  ln -s $snli/dev.r $out/dev.r
  ln -s $snli/test.l $out/test.l
  ln -s $snli/test.r $out/test.r
  ln -s $dataset/train.qrels $out/train.qrels
  ln -s $dataset/dev.qrels $out/dev.qrels
  ln -s $dataset/test.qrels $out/test.qrels
}

plan HypOnly {
  reach hypOnlyDataset via (Mode: *)
}
