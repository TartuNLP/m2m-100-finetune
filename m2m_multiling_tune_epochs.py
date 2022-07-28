#!/usr/bin/env python

import sys

import json

from tok import get_extended_m2m100_tokenizer as gettok

from random import shuffle

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, M2M100Tokenizer
from transformers.tokenization_utils_base import BatchEncoding

from datasets import load_dataset, load_metric, concatenate_datasets

from datetime import datetime

def log(msg):
	print(str(datetime.now()) + ": " + str(msg))

def getlangs(fn):
	with open(fn, 'r', encoding='utf-8') as fh:
		firstline = fh.readline().strip()
		
		langs = sorted(json.loads(firstline)['translation'].keys())
		
		if langs[0] + "-" + langs[1] in fn:
			return langs
		elif langs[1] + "-" + langs[0] in fn:
			return [langs[1], langs[0]]
		else:
			raise Exception("Filename should include the language pair in the form of xx-yy")

def prep_one_dataset(filename, tokenizer, mdlid):
	#ready_data = load_from_cache(filename, mdlid)
	
	#if not ready_data:
	
	dataset = load_dataset("json", data_files=filename )
	
	srcl, tgtl = getlangs(filename)
	
	tokenizer.src_lang = srcl
	tokenizer.tgt_lang = tgtl
	
	def preproc_func(examples):
		ins = [ex[srcl] for ex in examples['translation']]
		outs = [ex[tgtl] for ex in examples['translation']]
		
		result = tokenizer(ins, max_length=128, padding=True, truncation=True)
		
		with tokenizer.as_target_tokenizer():
			labels = tokenizer(outs, max_length=128, padding=True, truncation=True)
		
		result['labels'] = labels['input_ids']
		
		return result
	
	ready_data = dataset['train'].map(preproc_func, batched=True, desc="preproc files", remove_columns=['translation'])
	
	return ready_data

def prepdata(filenames, tokenizer, mdlid):
	filelist = filenames.split(":")
	
	datasets = [prep_one_dataset(f, tokenizer, mdlid) for f in filelist]
	
	meta = [ (f, d.num_rows) for f, d in zip(filelist, datasets) ]
	
	return concatenate_datasets(datasets), meta

def get_trainer(tok, mdl, trainset, devset, devmeta, outdir, batch_size = 12, gradient_accumulation_steps = 8, learning_rate = 5e-05, weight_decay = 0.00, num_epochs = 10):
	args = Seq2SeqTrainingArguments(
		 outdir,
		 evaluation_strategy = "epoch",
		 save_strategy = "epoch",
		 learning_rate=learning_rate,
		 per_device_train_batch_size=batch_size,
		 per_device_eval_batch_size=batch_size,
		 weight_decay=weight_decay,
		 gradient_accumulation_steps=gradient_accumulation_steps,
		 save_total_limit=None,
		 num_train_epochs=num_epochs,
		 predict_with_generate=True,
                 logging_dir='logs'   
	)
	
	data_collator = DataCollatorForSeq2Seq(tok, model=mdl)
	
	metric = load_metric("sacrebleu")
	
	def compute_metrics(eval_preds):
		hyp, ref = eval_preds
		if isinstance(hyp, tuple):
			hyp = hyp[0]
		
		dechyp = [pr.strip() for pr in tok.batch_decode(hyp, skip_special_tokens=True)]
		decref = [[hp.strip()] for hp in tok.batch_decode(ref, skip_special_tokens=True)]
		
		currStart = 0
		result = {}
		for filename, rownum in devmeta:
			metrresult = metric.compute(predictions=dechyp[currStart:currStart+rownum], references=decref[currStart:currStart+rownum])
			keyname = "bleu_" + filename
			result[keyname] = metrresult['score']
			currStart += rownum
		
		return result
	
	return Seq2SeqTrainer(
		mdl,
		args,
		train_dataset=trainset,
		eval_dataset=devset,
		data_collator=data_collator,
		tokenizer=tok,
		compute_metrics=compute_metrics
	)

def loadextras(filename):
	with open(filename, "r", encoding='utf-8') as fh:
		return [t.strip() for t in fh]

def loadtok(initmdl):
	extratoks = loadextras("FINNO-UGRIC_2022/m2mtune/extra_tokens.txt")
	
	result = gettok(initmdl, "liv vro sma sme smn sms smj".split(), extratoks)
		
	return result

def loadmdl(initmdl, newnum):
	result = AutoModelForSeq2SeqLM.from_pretrained(initmdl)
	
	result.resize_token_embeddings(newnum)
	
	return result

if __name__ == "__main__":
	_, initmdl, outdir, trainfiles, devfiles, num_epochs = sys.argv
          	
	log("Loading tokenizer")
	tokenizer = loadtok(initmdl)
	
	log("Loading data")
	traindata, _ = prepdata(trainfiles, tokenizer, initmdl)
	
	log("Loading dev")
	devdata, meta = prepdata(devfiles, tokenizer, initmdl)
	
	log("Loading model")
	model = loadmdl(initmdl, len(tokenizer))
	
	log("Start training")
	trainer = get_trainer(tokenizer, model, traindata, devdata, meta, outdir, num_epochs = int(num_epochs))
	
	log("Starting training")
	trainer.train()

	log("Saving model")
	trainer.save_model(outdir)
	
	log("Done!")
