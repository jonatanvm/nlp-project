import fasttext
import sys
import uuid

file_name = sys.argv[1] if len(sys.argv) >= 2 else None

# help(fasttext.FastText)

if not file_name:
    # Parameters: lr=1.0, epoch=25, wordNgrams=2, bucket=200000, dim=50, loss='hs/ova/one-vs-all'
    model = fasttext.train_supervised(input="models/fast_text/speeches.train", lr=1.0, epoch=25, wordNgrams=3)
    model.save_model(f"models/fast_text/model_speeches-{uuid.uuid4().hex[:6]}.bin")
else:
    model = fasttext.FastText.load_model(f"models/fast_text/{file_name}")
    print(f"Loaded model: {file_name}")
n_samples, precision, recall = model.test("models/fast_text/speeches.test")

print(f"precision: {precision}, recall: {recall}")
