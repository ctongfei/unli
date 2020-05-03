task regression : unli
  < dataset=(RegressionScenario:
    surrogate=$out@snliWithSurrogatesDataset
    combined=$out@snliCombinedWithUSnliDataset
    usnli=$out@uSnliDataset
    hyp-only-surrogate=$out@hypOnlyDataset[Mode:surrogate]
    hyp-only-combined=$out@hypOnlyDataset[Mode:combined]
    hyp-only-usnli=$out@hypOnlyDataset[Mode:fineTune]
  )
  > out
  :: .submitter=@ .action_flags=@ .resource_flags=@
  :: gpuId=@
{
  mkdir -p $out
  PYTHONPATH=$unli python $unli/unli/commands/regression.py \
    --data $dataset \
    --out $out
}

plan Regression {
  reach regression via (RegressionScenario: *)
}
