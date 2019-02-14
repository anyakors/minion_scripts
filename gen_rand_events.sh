for N in {1..512}
do
	echo "Channel_$N"
	python gen_random_events.py -f ~/workspace/minion/models/mookse_Veriton_X4650G_20180613_FAH54029_MN21778_sequencing_run_RNA3_G4_false_8_78820.fast5 -s ~/workspace/minion/selected/random/ -c "Channel_$N"
done