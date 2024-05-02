#!/bin/bash
WDIR=`realpath -s $1`  # working folder
IN=$WDIR/seq.fasta
export PETFOLDBIN="/expanse/projects/itasser/jlspzw/nwentao/ss-program/PETfold/bin"
PYTHON="python"
export MKL_THREADING_LAYER=GNU
full_path="/expanse/projects/itasser/jlspzw/nwentao/projects/DRfold/DRfold.sh"
 
dir_path=$(dirname $full_path)

if [ ! -s $WDIR/DPRcg.pdb ]
then
    echo "Optimizing structure"
    $PYTHON $dir_path/PotentialFold/Fold.py $(realpath $IN) $WDIR/geo.npy $WDIR/DPRcg $WDIR/e2e_0.npy $WDIR/e2e_1.npy $WDIR/e2e_2.npy $WDIR/e2e_3.npy $WDIR/e2e_4.npy $WDIR/e2e_5.npy  >>$WDIR/running.log
fi

if [ ! -s $WDIR/DPR_5.pdb ]
then
    echo "Filling full atoms"
    $dir_path/bin/Arena $WDIR/DPRcg_5.pdb $WDIR/DPR_5.pdb 6 >>$WDIR/running.log
fi
if [ ! -s $WDIR/DPR_4.pdb ]
then
    echo "Filling full atoms"
    $dir_path/bin/Arena $WDIR/DPRcg_4.pdb $WDIR/DPR_4.pdb 6 >>$WDIR/running.log
fi
if [ ! -s $WDIR/DPR_3.pdb ]
then
    echo "Filling full atoms"
    $dir_path/bin/Arena $WDIR/DPRcg_3.pdb $WDIR/DPR_3.pdb 6 >>$WDIR/running.log
fi
if [ ! -s $WDIR/DPR_2.pdb ]
then
    echo "Filling full atoms"
    $dir_path/bin/Arena $WDIR/DPRcg_2.pdb $WDIR/DPR_2.pdb 6 >>$WDIR/running.log
fi
if [ ! -s $WDIR/DPR_1.pdb ]
then
    echo "Filling full atoms"
    $dir_path/bin/Arena $WDIR/DPRcg_1.pdb $WDIR/DPR_1.pdb 6 >>$WDIR/running.log
fi
if [ ! -s $WDIR/DPR_0.pdb ]
then
    echo "Filling full atoms"
    $dir_path/bin/Arena $WDIR/DPRcg_0.pdb $WDIR/DPR_0.pdb 6 >>$WDIR/running.log
fi

if [ ! -s $WDIR/DPRr1.pdb ]
then
    echo "Filling full atoms"
    $dir_path/bin/Arena $WDIR/DPRcg.pdb $WDIR/DPRr1.pdb 6 >>$WDIR/running.log
fi

if [ ! -s $WDIR/DPR.pdb ]
then
    echo "Refinement"
    $PYTHON $dir_path/scripts/refine.py $WDIR/DPRr1.pdb $WDIR/DPR.pdb 0.6 >>$WDIR/running.log
fi