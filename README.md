python main.py \
  --train_pickle "/Users/john/Desktop/data/pickle/combined.pickle" \
  --test_pickle  "/Users/john/Desktop/data/pickle/3GC_test.pickle" \
  --seq_len 64 --batch_size 32 --epochs 12 --lr 5e-4 \
  --model transformer \
  --out_csv transformer_final_test.csv

!python /content/transformerv2/main.py \
  --train_pickle "/content/transformerv2/combined_v3.pickle" \
  --test_pickle  "/content/transformerv2/3GC_normalized.pickle" \
  --sequence_length 64 --batch_size 32 --epochs 2 --lr 5e-4 \
  --model transformer \
  --out_csv transformer_final_test.csv

!python /content/transformerv2/run_test_only.py \
  --test_pickle "/content/transformerv2/5GC_normalized.pickle" \
  --model_path "/content/best_model.pt" \
  --output_csv transformer_test_5GC.csv \
  --model transformer
