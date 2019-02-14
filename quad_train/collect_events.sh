for N in {1..6}
do
	echo "Channel_$N"
	python collect_strand_reads.py -f /media/mookse/DATA1/minion_data/bulk/mookse_Veriton_X4650G_20180614_FAH54029_MN21778_sequencing_run_RNA9_mut_55737.fast5 -s /home/mookse/workspace/MinKNOW/training_sets/quad_noquad/noquad -c "Channel_$N"
done