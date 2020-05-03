global {

  aggregator=(Aggregator: mean median)
  EXTRACTOR_TYPE=(ExtractorType: decoupled)
  EMBEDDING_TYPE=(EmbeddingType: uncontextualized elmo bert)
  ENCODER_TYPE=(EncoderType: bilstmmax bertconcat)
  LOSS_TYPE=(LossType: pointwise-l2 pointwise-log pairwise-hinge)
  ENCODER_DIM=(EncoderDim: 1024)
  MARGIN=(Margin: 0.3 0.35 0.4 0.6 0.7 0.8)
  CRR_WEIGHT=(CrrWeight: 0.0837 0.0419 0.00837 0.00419)
  CROSS_SAMPLE=(CrossSample: 1 2)

}
