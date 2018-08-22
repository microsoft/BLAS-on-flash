#!/bin/bash
# INSTRUCTIONS:
# * run using `bash gemm_run.sh | grep 'fail\|max-relative\|CONFIG'` to check which test cases are failing
DIM=3072
MAT_DIR=/raid/matrices/dense

function run_config
{
  IN_A=${MAT_DIR}/${DIM}_${DIM}_A.bin
  IN_B=${MAT_DIR}/${DIM}_${DIM}_B.bin
  OUT_C_MEM=${MAT_DIR}/${DIM}_${DIM}_C.bin
  OUT_C_FLASH=${MAT_DIR}/${DIM}_${DIM}_C.bin2
  echo -e "CONFIG:$1"
  for i in {1..10};
  do
    fallocate -l $(($DIM * $DIM * 4)) ${OUT_C_FLASH}
    fallocate -l $(($DIM * $DIM * 4)) ${OUT_C_MEM}
    # generate input matrices
    # NOTE: USE ../bin/dense_create if available
    python -c "import numpy as np; a = np.random.rand(${DIM}, ${DIM}).astype(np.float32); a.tofile(\"${IN_A}\"); print(\"Generated input:${IN_A}\");"
    python -c "import numpy as np; a = np.random.rand(${DIM}, ${DIM}).astype(np.float32); a.tofile(\"${IN_B}\"); print(\"Generated input:${IN_B}\");"
    ../bin/in_mem_gemm_driver ${IN_A} ${IN_B} ${OUT_C_MEM} $DIM $DIM $DIM 1.0 0.0 $1 $DIM $DIM $DIM
    ../bin/gemm_driver ${IN_A} ${IN_B} ${OUT_C_FLASH} $DIM $DIM $DIM 1.0 0.0 $1 $DIM $DIM $DIM
    python -c "import numpy as np; a = np.fromfile(\"${OUT_C_FLASH}\", dtype=np.float32); b = np.fromfile(\"${OUT_C_MEM}\", dtype=np.float32); a=a[:b.shape[0]];print(\"max-relative-error=\" + str(np.max(np.divide(np.abs(a-b), b))))"
  done
	# cleanup
	rm ${IN_A} ${IN_B} ${OUT_C_MEM} ${OUT_C_FLASH}
}

# run all 8 configs
run_config "N N R"
run_config "N N C"
run_config "N T R"
run_config "N T C"
run_config "T N R"
run_config "T N C"
run_config "T T R"
run_config "T T C"
