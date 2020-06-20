from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

reference = [['the', 'cat',"is","sitting","on","the","mat"], ["cat", "on", "a", "mat"]]
test = [ "cat", "sitting", "on", "a", "mat"]
score = sentence_bleu(  reference, test, weights=(.5,.5, 0, 0))
print("example 1:", score)


references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
candidates = [['this', 'is', 'a', 'test']]
score = corpus_bleu(references, candidates)
print("example 2", score)


reference = [['the', 'cat',"is","sitting","on","the","mat"], ["cat", "on", "a", "mat"]]
test = [ "fast", "red", "car"]
score = sentence_bleu(  reference, test, weights=(0,1, 0, 0))
print("example 3:", score)



references = [['this', 'is',  'samell', 'a','test']]
candidates = ['this', 'is',  'a', 'test']

score1 = sentence_bleu(references, candidates, weights=(1,0,0,0))
score2 = sentence_bleu(references, candidates, weights=(0,1,0,0))
score3 = sentence_bleu(references, candidates, weights=(0,0,1,0))
score4 = sentence_bleu(references, candidates, weights=(0,0,0,1))

print( "%.2f %.2f %.2f %.2f" % (score1, score2, score3, score4))

score = sentence_bleu(references, candidates)
print(score)



# one word different 
# from nltk.translate.bleu_score import sentence_bleu 
reference = [[' the' , ' quick' , ' brown' , ' fox' , ' jumped' , ' over' , ' the' , ' lazy' , ' dog' ]]
candidate = [' the' , ' fast' , ' brown' , ' fox' , ' jumped' , ' over' , ' the' ] #, ' sleepy' , ' dog' ] 
score = sentence_bleu(reference, candidate)
print(score)