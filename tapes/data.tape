global {

  snliUrl="https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
  uSnliUrl="http://nlp.jhu.edu/unli/u-snli.zip"

}


task snli
  :: snliUrl=@
  > out
{
  mkdir -p $out
  wget $snliUrl
  unzip snli_1.0.zip
  mv snli_1.0/* $out
  rm -r snli_1.0
}

task uSnliCsv
  :: uSnliUrl=@
  > out
{
  mkdir -p $out
  wget $uSnliUrl
  unzip u-snli.zip
  mv *.csv $out
}

task snliQRels : unli
  < snli=$out@snli
  > out
{
  mkdir -p $out
  python $unli/scripts/snli_to_qrels.py \
    --snli $snli/snli_1.0 \
    --out $out
}

task uSnliQRels : unli
  < snliQRels=$out@snliQRels
  < uSnliPath=$out@uSnliCsv
  > out
{
  mkdir -p $out
  python $unli/scripts/usnli_to_qrels_align.py \
    --snli $snliQRels \
    --usnli_train $uSnliPath/train.csv \
    --usnli_dev $uSnliPath/dev.csv \
    --usnli_test $uSnliPath/test.csv \
    --out $out
}

