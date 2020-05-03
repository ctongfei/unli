## Uncertain Natural Language Inference

This repository hosts the code for the following paper:
 * Tongfei Chen*, Zhengping Jiang*, Adam Poliak, Keisuke Sakaguchi, Benjamin Van Durme (2020): 
   Uncertain natural language inference. In _Proceedings of ACL_.

### Preprequisites
 * Python >= 3.6

### Running

This repository uses [Ducttape](https://github.com/jhclark/ducttape) to manage intermediate results 
of the experiment pipeline.

To run a portion of the pipeline, use the following command:

  ```bash
    ducttape unli.tape -p <TASK>
  ```
  where `<TASK>` is any of the following:
  
| Task         | Description                                                      |
|--------------|------------------------------------------------------------------|
| `Data`       | Prepares SNLI and u-SNLI datasets (automatically downloads data) |
| `HypOnly`    | Generates datasets for hypothesis-only baselines                 |
| `Regression` | Trains the regression model under various conditions             |

One can easily execute different tasks by modifying the plans in the tape files.

### Citation
Please cite this paper and package as
```bibtex
@inproceedings{UNLI-ACL20,
    author = {Tongfei Chen and Zhengping Jiang and Adam Poliak and Keisuke Sakaguchi and Benjamin {Van Durme}},
    title = {Uncertain natural language inference},
    booktitle = {Proceedings of ACL},
    year = {2020}
}
```
