#!/bin/bash

params=(
"20 6a5f3d287fb1b0a6 c1a0679bd25f06c5"
"21 0ac3cb45892c96d6 cb28af67fc74dcbc"
"22 cd842501442e84f0 63778751f0e4bedb"
"23 1a3bd0f512dba4fd 1706a8b7d71f3017"
"24 e3b8fb27d90bd863 0c094fccd755cde2"
"25 dea7744bf2755ad9 6e0ebdc050a492ee"
"26 18667086664cac57 7ceb2e54e389be66"
"27 ce2c621789812038 13950eec4d6caf2b"
"28 0094a07e0d20f0d4 3e29df756df6b567"
"29 19b05757e5e766a4 09a83a6bee5132a1"
"30 6b588b953555fdc3 857f62bb3ab86c48"
"31 9cbcfd09ac384207 b486d19a8ebf1afa"
"32 3458103b639f9f47 9e1499d15be61572"
"33 4d1b29953931a24f e0bc6cefe332ac33"
"34 9fb679b7ce303683 adf075f18af8e8fb"
"35 ce7edd87642dc9dd 9a192c412800b2c5"
"36 2ddbc9fe4da17416 fab28111310cbea3"
)

for p in "${params[@]}"; do
    
    n=$(echo $p | awk '{print $1}')
    C0=$(echo $p | awk '{print $2}')
    C1=$(echo $p | awk '{print $3}')

        echo "Soumission pour --n $n --C0 $C0 --C1 $C1"
    
        mpiexec --mca pml ob1 --mca btl tcp,self --hostfile $OAR_NODEFILE -n $(wc -l < $OAR_NODEFILE) final --n "$n" --C0 "$C0" --C1 "$C1"
done
