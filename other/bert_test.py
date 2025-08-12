from evaluate import load


bertscore = load("bertscore")

cands = ["Acetaminophen"]
refs = ["Ibuprofen"]

results1 = bertscore.compute(predictions=cands, references=refs, model_type="distilbert-base-uncased")

print(results1["precision"][0],results1["recall"][0],results1["f1"][0])
