from flask import Flask, render_template, request, send_file
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
import gensim
import numpy as np
import random
import pandas as pd
from nltk.tokenize import word_tokenize
import os.path
import argparse

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/accuracy_prediction', methods= ['POST'])
def accuracy_prediction():
	global Filename
	model = gensim.models.doc2vec.Doc2Vec.load('doc2vec_debugmode.d2v')
	if request.method == "POST":
		file = request.files['file']
		myFile = file.filename
		print(myFile)
		try:
			with open(myFile) as f:
				text = []
				for line in f.readlines():
					text.append(line)
			f.close()

			tag = []
			question = []
			accuracy = []

			for Question in text:
				data = word_tokenize(Question)
				arr = np.asarray(data)
				docvec = model.infer_vector(arr)
				sims = model.docvecs.most_similar([docvec], topn= 1)
				if str(sims[0][0]).startswith('Q') and sims[0][1] >= 0.85:
					tag.append(sims[0][0])
					temp = Question.strip().split("\n")
					question.append(temp[0])
					accuracy.append(sims[0][1])

					with open(myFile+'SA','a') as f:
						f.write('Tag: {} '.format(sims[0][0]))
						f.write('Accuracy: {} '.format(sims[0][1]))
						f.write('Question: {}'.format(Question))
						f.write('\n')

			Df = {'Tag': tag, 'Question': question, 'Accuracy': accuracy}
			Df = pd.DataFrame(data= Df)
			Filename = "Accuracy_Predictions.csv"
			Df.to_csv(Filename, index= None)
			return render_template("index.html", text= Df.to_html(), btn='download.html')

		except Exception as e:
			return render_template("index.html", text= str(e))

@app.route("/download-file/")
def download():
	return send_file(Filename, attachment_filename= 'Output.txt', as_attachment=True)

if __name__ == "__main__":
	app.run(debug= True)