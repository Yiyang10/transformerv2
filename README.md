python main.py \
  --train_pickle "/Users/john/Desktop/data/pickle/combined.pickle" \
  --test_pickle  "/Users/john/Desktop/data/pickle/3GC_test.pickle" \
  --seq_len 64 --batch_size 32 --epochs 12 --lr 5e-4 \
  --model transformer \
  --out_csv transformer_final_test.csv
