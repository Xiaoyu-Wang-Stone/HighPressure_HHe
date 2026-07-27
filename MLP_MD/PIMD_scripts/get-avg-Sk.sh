abnsys=$1

for prefix in HeHe HeH HH; do

for a in `seq 1 64`; do 
awk -v a=$a '!/#/{print $a}' Sk${sys}-${prefix}-real.dat | /global/scratch/users/xiaoyuwang/software/toolbox/bin/autocorr -maxlag 20 | head -n 1 | awk '{print $2,$3,$6}'; 
done > $prefix-tmp

paste Sk$sys-kgrid.dat $prefix-tmp  > Sk${sys}-${prefix}-real-avg.list
rm $prefix-tmp

done
