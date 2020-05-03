package unli
    :: .versioner=disk
    :: .path="/home/tongfei/proj/unli-release"
    :: .rev=NULL
{}

global {

  ducttape_output="out"
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true

  .submitter="shell"
  .action_flags=(Device:
    cpu=""
    gpu=""
  )
  .resource_flags=(Device:
    cpu=""
    gpu=""
  )
  gpuId="0"

}

import "tapes/sge.tape"
import "tapes/params.tape"
import "tapes/data.tape"
import "tapes/assemble.tape"
import "tapes/hyponly.tape"
import "tapes/regression.tape"
