python train.py --dataset Cora --method recoverability > core_recoverability.txt &
sleep 30s
python train.py --dataset Cora --method GRACE > core_grace.txt &
sleep 30s
python train.py --dataset CiteSeer --method recoverability > citeseer_recoverability.txt &
sleep 30s
python train.py --dataset CiteSeer --method GRACE > citeseer_grace.txt &
sleep 30s
python train.py --dataset PubMed --method recoverability > pubmed_recoverability.txt &
sleep 30s
python train.py --dataset PubMed --method GRACE > pubmed_grace.txt &
sleep 30s
python train.py --dataset DBLP --method recoverability > dblp_recoverability.txt &
sleep 30s
python train.py --dataset DBLP --method GRACE > dblp_grace.txt &
sleep 30s