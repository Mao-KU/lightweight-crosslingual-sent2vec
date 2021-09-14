readonly var=$4
la=$5

mkdir -p $3
if "${var}"; then
	CUDA_VISIBLE_DEVICES=$1 nohup python main.py -m $2 -r True -la $la >> $3/$2.out 2>&1 &
	echo "CUDA_VISIBLE_DEVICES=$1 nohup python main$5.py -m $2 -r True -la $la >> $3/$2.out 2>&1 &"
else
	CUDA_VISIBLE_DEVICES=$1 nohup python main.py -m $2 -la $la >> $3/$2.out 2>&1 &
	echo "CUDA_VISIBLE_DEVICES=$1 nohup python main$5.py -m $2 -la $la >> $3/$2.out 2>&1 &"
fi

