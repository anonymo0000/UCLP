##1.Notes:
Code and data are placed separately, with all code in the code folder and all data in the data folder. 
It should be noted that due to the large number of experiments in this article, the data folder is large and stores the results of all experiments.

##2.Key Aspect Extract:
This module uses regular expression rules to extract descriptions corresponding to six key aspects from the vulnerability report text. 
The input is the original vulnerability report text description data (data/allitem_v5. csv), and the output is the corresponding descriptions for each key aspect.
Run code/get_4aspects/module/pt_. py in sequence, code/get_4aspects/module/pt2_.py，code/get_4aspects/module/pt3_.py. 
The rules in each file can only process a portion of the data. Output the data that cannot be processed in this file to an intermediate file, which serves as the input for the next file.
Finally, merge the outputs of the three files to obtain all the extracted key aspects.
The specific steps are as follows:
1) Run pt1 _. py, input as data/allitem_v5.csv, output the data that can be extracted as the extraction result r1, and output the data d1 that cannot be processed;
2) Run ptd_. py, input d1, output the data that can be extracted as the extraction result r2, and output the data d2 that cannot be processed;
3) Run pt3_. py, input d2, output the data that can be extracted as the extraction result r3, and output the data d3 that cannot be processed;
4) Merge r1, r2, and r3 as the extraction results.
This section originates from https://github.com/pmaovr/Predicting-Missing-Aspects-of-Vulnerability-Reports

##3.Classification
This method classifies the extracted key aspects, with the input being the corresponding description of the key aspects and the output being category labels.
Run: /module/get_att_type_class_multiThread_for_vtype.ipynb; and /home/user1/lh/PMA_lihang/mypaper/module/tf-idf-multiThread copy 2.ipynb

##4.Based on attention capture semantics
The core code is: ''aspect2=torch. mean (aspect [: 1], 0) * 0.5+torch. mean (aspect [1:], 0) * 0.5'' to capture attention to the semantics of the first few words in the key aspect description text.

##5.Introducing Topic Words to Capture Domain Semantics of Tags
Running files
code/tempp.ipynb
Obtain the definitions of each tag from wordNet and select the definition with the highest similarity to the theme word.

##6.Hyperparameter adaptive mechanism
The core code is: if max_sim<0.6 to control the similarity threshold, use outputs under different thresholds, and observe the directory
code/model
Compare the classification performance of different models (CNN, BERT, etc.) to obtain the optimal threshold configuration.

##7.UCLP_module1(label profile):
Run:/module/divide3Context_att_type.ipynb
Using key aspect classification methods to obtain annotated data, construct class documents corresponding to each label.

##8.UCLP_module2(Jonit TF-IDF):
The core code is :"score_0 = 1 * (get_weight(sorted_word2tfidf0_v,set_v0,v6) * 5 + get_weight(sorted_word2tfidf0_i,set_i0,i6) * 4 + get_weight(sorted_word2tfidf0_t,set_t0,t6) * 3)".
The code before and after this line weights different key aspects in the context of missing aspects to calculate the semantic similarity between the context and the class document.

##9.UCLP_module3(Smoothing Function):
The core code is: "score+=math. pi/2- math. atan ((1e-8) * (index))", 
which maps the TF-IDF ranking of a word using the inverse cotangent function to obtain the semantic similarity score between the word and the class document.

##10.requirement:
numpy==1.23.5
pandas==2.0.3
nltk==3.8.1
spacy==3.5.3
transformers==4.29.0
gensim==4.3.1
tensorflow==2.12.0
torch==2.0.1
scikit-learn==1.3.0
tqdm==4.65.0
matplotlib==3.7.1
seaborn==0.12.2
jupyterlab==4.0.4
notebook==7.0.0
