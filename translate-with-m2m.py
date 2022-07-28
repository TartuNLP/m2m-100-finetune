#!/usr/bin/env python

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from m2m_multiling_tune import loadtok

import sys
import re
import torch
import os

from datetime import datetime

def log(msg):
	print(str(datetime.now()) + ": " + msg, file = sys.stderr)

def tokenize(tok, snt, srcLang):
	tok.src_lang = srcLang
	return tok(snt, return_tensors="pt", padding=True).to("cuda")
	#return tok(snt, return_tensors="pt", padding=True)

def translate(mdl, tok, enctxt, trgLang):
	generated_tokens = mdl.generate(**enctxt, forced_bos_token_id=tok.get_lang_id(trgLang))

	return tok.batch_decode(generated_tokens, skip_special_tokens=True)

def loadmdl(mdl):
	log("Load model")
	model = M2M100ForConditionalGeneration.from_pretrained(mdl)
	
	log("Model to GPU")
	model.to("cuda")
	return model

def fixit(self):
	self.id_to_lang_token = dict(list(self.id_to_lang_token.items()) + list(self.added_tokens_decoder.items()))
	self.lang_token_to_id = dict(list(self.lang_token_to_id.items()) + list(self.added_tokens_encoder.items()))
	self.lang_code_to_token = { k.replace("_", ""): k for k in self.additional_special_tokens }
	self.lang_code_to_id = { k.replace("_", ""): v for k, v in self.lang_token_to_id.items() }

def m2mTranslate(srcList, srcLang, tgtLang):
	enc = tokenize(tokenizer, srcList, srcLang)
	out = translate(model, tokenizer, enc, tgtLang)
	return out

def getLpFromFile(filename):
	langsInFile = []
	filename = os.path.basename(filename)
	splits = re.split(r"[.-]", filename)
	
	for l in "et lv fi vro liv sma sme en no smn sms smj".split():
		for l2 in splits:
			if l == l2:
				langsInFile += [l]
	
	if len(langsInFile) != 2:
		print(filename)
		print(splits)
		print(langsInFile)
		print("Choked on " + filename + " :-(")
		raise Exception
	
	if (langsInFile[0] + "-" + langsInFile[1]) in filename:
		return langsInFile[0], langsInFile[1]
	elif (langsInFile[1] + "-" + langsInFile[0]) in filename:
		return langsInFile[1], langsInFile[0]
	else:
		print("Still choked on " + filename + " :-((")

def getOutFile(filename, suffix, mdlname):
	basename = filename.split("/")[-1]
	basetrunc = ".".join(basename.split(".")[:-1])
	
	return mdlname + "/" + basetrunc + "." + suffix

def partition(lines, chunkSize = 50):
	linesCopy = lines
	
	while (linesCopy):
		yield(linesCopy[:chunkSize])
		linesCopy = linesCopy[chunkSize:]

def translateFile(srcLang, tgtLang, infile, outfile, translFunc, chunk = 50):
	with open(infile, 'r', encoding='utf-8') as fh:
		lines = [line.strip() for line in fh]
		
	with open(outfile, 'w', encoding='utf-8') as ofh:
		for chunk in partition(lines, chunkSize = chunk):
			trs = translFunc(chunk, srcLang, tgtLang)
			
			try:
				for tr in trs:
					print(tr, file=ofh)
			except TypeError as e:
				sys.stderr.write("AAA: {0}\n".format(len(" ".join(chunk))))
				raise e

def translateFh(srcLang, tgtLang, translFunc, chunk = 50, fh = sys.stdin):
	lines = [line.strip() for line in sys.stdin]
	
	for chunk in partition(lines, chunkSize = chunk):
		trs = translFunc(chunk, srcLang, tgtLang)
		
		try:
			for tr in trs:
				print(tr)
		except TypeError as e:
			sys.stderr.write("AAA: {0}\n".format(len(" ".join(chunk))))
			raise e

if __name__ == "__main__":
	mdlname = sys.argv[1]
	
	log("Loading tokenizer")
	tokenizer = loadtok(mdlname)
	
	model = loadmdl(mdlname)
	
	if len(sys.argv) == 3:
		lp = sys.argv[2]

		(srcLang, tgtLang) = lp.split("-")
		
		log("Start translating")
		translateFh(srcLang, tgtLang, translFunc = m2mTranslate, chunk = 8)
		log("Done")
	else:
		suffix = sys.argv[2]
		
		for filename in sys.argv[3:]:
			srcLang, tgtLang = getLpFromFile(filename)
			outfile = getOutFile(filename, suffix, mdlname)
			log("Translating " + filename + " into " + outfile + " from " + srcLang + " into " + tgtLang)
			translateFile(srcLang, tgtLang, filename, outfile, m2mTranslate, chunk = 8)
