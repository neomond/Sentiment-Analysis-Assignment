#!/usr/bin/env python
import re, random, math, collections, itertools

PRINT_ERRORS=0 #Change from 0 to 1

#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())
 
#  CODE MODIFICATION (Step 2.1): added filtering logic for dictionary parsing
    posDictionary = open('positive-words.txt', 'r', encoding="ISO-8859-1")
    posWordList = [] # editting this to get the list of the positive sentiment words in the dictionary
    for line in posDictionary:
        line = line.strip()
        if line and not line.startswith(';'):
            posWordList.append(line)
# END MODIFICATION            

    negDictionary = open('negative-words.txt', 'r', encoding="ISO-8859-1")
    negWordList = [] # editting this to get the list of the negative sentiment words in the dictionary
    for line in negDictionary:
        line = line.strip()
        if line and not line.startswith(';'):
            negWordList.append(line)
# END MODIFICATION              

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    # Create Training and Test Datsets:
    # We want to test on sentences we haven't trained on, 
    # to see how well the model generalses to previously unseen sentences

    # create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    # create Nokia Dataset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#

# calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    # iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: # calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                # keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1 # keeps count of total words in negative class
                
                # keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        # do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

# implement naive bayes algorithm
# INPUTS:
#   sentencesTest is a dictonary with sentences associated with sentiment 
#   dataName is a string (used only for printing output)
#   pWordPos is dictionary storing p(word|positive) for each word
#      i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#   pWordNeg is dictionary storing p(word|negative) for each word
#   pWord is dictionary storing p(word)
#   pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):

    print("Naive Bayes classification")
    pNeg=1-pPos

    # These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    # for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: # calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
 
 
# TODO for Step 2: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
#CODE MODIFICATION (Step 2.2): Added evaluation metrics calculation
    # 1. Accuracy
    accuracy = correct / total

    # 2. Precision for positive class
    # Precision = True Positives / (True Positives + False Positives)
    # correctpos = True Positives
    # totalpospred = all predicted as positive
    if totalpospred > 0:
        precision_pos = correctpos / totalpospred
    else:
        precision_pos = 0

    # 3. Recall for positive class
    # Recall = True Positives / (True Positives + False Negatives)
    # correctpos = True Positives
    # totalpos = all actual positives    
    if totalpos > 0:
        recall_pos = correctpos / totalpos
    else:
        recall_pos = 0
        
    # 4. Precision for negative class
    if totalnegpred > 0:
        precision_neg = correctneg / totalnegpred
    else:
        precision_neg = 0

    # 5. Recall for negative class
    if totalneg > 0:
        recall_neg = correctneg / totalneg
    else:
        recall_neg = 0

    # 6. F1 Score (for positive class)
    if (precision_pos + recall_pos) > 0:
        f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)
    else:
        f1_pos = 0

    # Print results
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Positive - Precision: {precision_pos:.4f}, Recall: {recall_pos:.4f}")
    print(f"Negative - Precision: {precision_neg:.4f}, Recall: {recall_neg:.4f}")
    print(f"F1 Score: {f1_pos:.4f}")
    print()    
# END MODIFICATION


# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):

    print("Dictionary-based classification")
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total+=1
        if sentiment=="positive":
            totalpos+=1
            if score>=threshold:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1

# CODE MODIFICATION (Step 6.2): Added error printing to testDictionary
                if PRINT_ERRORS:
                    print(f"ERROR (pos classed as neg, score={score}): {sentence}")
# END MODIFICATION
        else:
            totalneg+=1
            if score<threshold:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1

                if PRINT_ERRORS:
                    print(f"ERROR (neg classed as pos, score={score}): {sentence}")
 
    
# TODO for Step 5: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
# CODE MODIFICATION (Step 5.1): Added evaluation metrics to testDictionary
    accuracy = correct / total

    if totalpospred > 0:
        precision_pos = correctpos / totalpospred
    else:
        precision_pos = 0

    if totalpos > 0:
        recall_pos = correctpos / totalpos
    else:
        recall_pos = 0
        
    if totalnegpred > 0:
        precision_neg = correctneg / totalnegpred
    else:
        precision_neg = 0

    if totalneg > 0:
        recall_neg = correctneg / totalneg
    else:
        recall_neg = 0

    if (precision_pos + recall_pos) > 0:
        f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)
    else:
        f1_pos = 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Positive - Precision: {precision_pos:.4f}, Recall: {recall_pos:.4f}")
    print(f"Negative - Precision: {precision_neg:.4f}, Recall: {recall_neg:.4f}")
    print(f"F1 Score: {f1_pos:.4f}")
    print()

# CODE MODIFICATION (Step 5.3) : New function - Improved rule-based system with linguistic rules
def testImprovedDictionary(sentencesTest, dataName, sentimentDictionary, threshold):
    print("Improved Dictionary-based classification")
    """
    Improved dictionary classifier with:
    - Negation handling (3-word window)
    - Intensifiers (multiply by 1.5)
    - Diminishers (multiply by 0.5)
    - Threshold changed from 1 to 0
    """
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    
    # Define negation words
    negation_words = ['not', 'no', 'never', 'nothing', 'neither', 'nobody', 'nowhere', 
                     'none', "n't", 'cannot', 'cant', 'dont', "don't", 'without']
    
    # Define intensifiers (increase sentiment)
    intensifiers = ['very', 'extremely', 'highly', 'absolutely', 'completely', 'totally',
                   'really', 'incredibly', 'amazingly', 'particularly', 'especially']
    
    # Define diminishers (decrease sentiment)
    diminishers = ['slightly', 'somewhat', 'barely', 'hardly', 'scarcely', 'little',
                  'bit', 'less', 'minor', 'marginally']
    
    for sentence, sentiment in sentencesTest.items():
        words = re.findall(r"[\w']+", sentence)
        words_lower = [w.lower() for w in words]
        score = 0
        
        for i, word in enumerate(words_lower):
            if word in sentimentDictionary:
                word_score = sentimentDictionary[word]
                
                # Check for negation in the 3 words before
                negated = False
                for j in range(max(0, i-3), i):
                    if words_lower[j] in negation_words:
                        negated = True
                        break
                
                # Check for intensifier right before
                intensified = False
                if i > 0 and words_lower[i-1] in intensifiers:
                    intensified = True
                
                # Check for diminisher right before
                diminished = False
                if i > 0 and words_lower[i-1] in diminishers:
                    diminished = True
                
                # Apply modifications
                if negated:
                    word_score = -word_score  # Flip the sentiment
                
                if intensified:
                    word_score = word_score * 1.5  # Increase by 50%
                
                if diminished:
                    word_score = word_score * 0.5  # Decrease by 50%
                
                score += word_score
 
        total += 1
        if sentiment == "positive":
            totalpos += 1
            if score >= threshold:
                correct += 1
                correctpos += 1
                totalpospred += 1
            else:
                totalnegpred += 1
        else:
            totalneg += 1
            if score < threshold:
                correct += 1
                correctneg += 1
                totalnegpred += 1
            else:
                totalpospred += 1
    
    # Calculate metrics (same as before)
    accuracy = correct / total
    
    if totalpospred > 0:
        precision_pos = correctpos / totalpospred
    else:
        precision_pos = 0
    
    if totalpos > 0:
        recall_pos = correctpos / totalpos
    else:
        recall_pos = 0
        
    if totalnegpred > 0:
        precision_neg = correctneg / totalnegpred
    else:
        precision_neg = 0
    
    if totalneg > 0:
        recall_neg = correctneg / totalneg
    else:
        recall_neg = 0
    
    if (precision_pos + recall_pos) > 0:
        f1_pos = 2 * (precision_pos * recall_pos) / (precision_pos + recall_pos)
    else:
        f1_pos = 0
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Positive - Precision: {precision_pos:.4f}, Recall: {recall_pos:.4f}")
    print(f"Negative - Precision: {precision_neg:.4f}, Recall: {recall_neg:.4f}")
    print(f"F1 Score: {f1_pos:.4f}")
    print()

# Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower[word]=1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print ("NEGATIVE:")
    print (head)
    print ("\nPOSITIVE:")
    print (tail)




#---------- Main Script --------------------------

# CODE MODIFICATION: Set random seed for reproducibility
random.seed(42)
# END MODIFICATION


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

# build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

# run naive bayes classifier on datasets
testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)



# run sentiment dictionary based classifier on datasets
testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1)
testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1)
testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1)

# Test improved dictionary
# print("\n=== IMPROVED RULE-BASED SYSTEM ===\n")
testImprovedDictionary(sentencesTrain,  "Films (Train Data, Improved)\t", sentimentDictionary, 0)
testImprovedDictionary(sentencesTest,  "Films  (Test Data, Improved)\t",  sentimentDictionary, 0)
testImprovedDictionary(sentencesNokia, "Nokia   (All Data, Improved)\t",  sentimentDictionary, 0)

# print most useful words
# mostUseful(pWordPos, pWordNeg, pWord, 100)

# CODE MODIFICATION (Step 4.2): Dictionary coverage analysis
# Q4.2: How many of these words are in the sentiment dictionary?
predictPower = {}
for word in pWord:
    if pWordNeg[word] < 0.0000001:
        predictPower[word] = 1000000000
    else:
        predictPower[word] = pWordPos[word] / (pWordPos[word] + pWordNeg[word])

sortedPower = sorted(predictPower, key=predictPower.get)
top_negative = sortedPower[:100]
top_positive = sortedPower[-100:]

neg_in_dict = sum(1 for word in top_negative if word in sentimentDictionary)
pos_in_dict = sum(1 for word in top_positive if word in sentimentDictionary)

print(f"\nNegative words in dictionary: {neg_in_dict}/100")
print(f"Positive words in dictionary: {pos_in_dict}/100")
# END MODIFICATION